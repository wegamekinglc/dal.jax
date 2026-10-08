"""Three-phase Longstaff-Schwartz training and frozen-policy replay."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from dal_jax.errors import DalError, InvalidPathCount, InvalidPayoff, script_error
from dal_jax.mc.engine import MonteCarloEngine, PathProduct
from dal_jax.mc.regression import Regression, _design, basis_powers, select_regression
from dal_jax.mc.regression_device import select_device
from dal_jax.mc.settings import MonteCarloSettings
from dal_jax.models.base import Model
from dal_jax.random import bridge
from dal_jax.random.inverse_normal import inverse_ncdf, inverse_ncdf_ndtri
from dal_jax.random.sobol import MAX_POINTS, MUL, Sobol, digital_shifts
from dal_jax.script.lower.lsmc import Records, _extend_state, lower_records, regression_scenario
from dal_jax.script.lower.smoothing import (
    cspr,
    cspr_bounds,
    smoothstep_cspr,
    smoothstep_cspr_bounds,
)
from dal_jax.script.lsmcprep import exercise_node

if TYPE_CHECKING:
    from dal_jax.script.preparation import PreparedProduct


class Policy(NamedTuple):
    coefficients: Array  # [events, max_basis]
    means: Array  # [events, features]
    sigmas: Array
    exercise_mask: Array

    def predict(self, event, x):
        coefficients, mean, sigma = self.coefficients[event], self.means[event], self.sigmas[event]
        constant = jnp.all(coefficients[1:] == 0.0)
        z = (jnp.where(constant, mean, x) - mean) / sigma
        if x.shape[-1] == 1:
            value = jnp.asarray(0.0, z.dtype)
            for coefficient in reversed(jnp.unstack(coefficients)):
                value = value * z[0] + coefficient
            return value
        degree = next(
            d for d in range(4) if len(basis_powers(x.shape[-1], d)) == coefficients.shape[-1]
        )
        return _design(z, basis_powers(x.shape[-1], degree)) @ coefficients


@dataclass(frozen=True, slots=True)
class TrainingResult:
    policy: Policy
    regressions: tuple[Regression, ...]


@dataclass(frozen=True, slots=True)
class LsmcResult:
    values: tuple[tuple[str, float], ...]
    replicate_means: tuple[float, ...]
    training: TrainingResult

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)


def _policy(regressions, prepared, degree):
    features = len(prepared.regression_features)
    width = len(basis_powers(features, degree))
    coefficients = np.zeros((len(prepared.events), width))
    means = np.zeros((len(prepared.events), features))
    sigmas = np.ones_like(means)
    mask = np.zeros(len(prepared.events), bool)
    for event, fit in zip(
        (i for i, row in enumerate(prepared.events) if exercise_node(row) is not None), regressions
    ):
        coefficients[event, : len(fit.coefficients)] = fit.coefficients
        means[event], sigmas[event], mask[event] = fit.means, fit.sigmas, True
    return Policy(*(jnp.asarray(field) for field in (coefficients, means, sigmas, mask)))


def scramble_key(pricing, seed, replicate=0):
    return (int(pricing) << 63) | (int(seed) << 32) | int(replicate)


def _path_counts(settings, n_paths):
    training = settings.lsmc_training_paths or n_paths
    validation = settings.lsmc_validation_paths or 0
    replicas = settings.lsmc_rqmc_replicates or 1
    if training + validation + replicas * n_paths > MAX_POINTS:
        raise InvalidPathCount(
            "LSMC training, validation and pricing replicas exceed the 32-bit Sobol sequence"
        )
    return training, validation, replicas


def _records_product(prepared, offset):
    width = len(prepared.events) * (4 + len(prepared.regression_features))
    errors = tuple(
        f"{code}: {name}"
        for name in prepared.variables.vector_names
        for code in ("VectorIndexOutOfRange", "EmptyVectorReduction")
    )

    def payoff(params, scenario, ctx, initial):
        records, _ = lower_records(prepared, ctx)(
            initial, regression_scenario(prepared, scenario), params["script"]
        )
        rows = jnp.concatenate(
            (
                records.payments[:, None],
                records.exercise_values[:, None],
                records.conditions[:, None],
                records.features,
                records.numeraires[:, None],
            ),
            axis=1,
        )
        return jnp.concatenate((rows.reshape(-1), records.errors.astype(rows.dtype)))

    initial = lambda params: _extend_state(
        prepared.initial_state(params), 3 + len(prepared.regression_features)
    )
    return PathProduct(
        timeline=prepared.timeline,
        sample_defs=prepared.sample_defs,
        script_params=prepared.script_params,
        payoff=payoff,
        payoff_names=tuple(str(i) for i in range(width)),
        initial_state=initial,
        error_messages=errors,
        path_offset=offset,
        path_state_size=len(prepared.initial_values)
        + sum(prepared.variables.vector_capacities)
        + 3
        + len(prepared.regression_features)
        + 3 * len(prepared.variables.vector_names),
    )


def _collect(engine, params, n_paths):
    values = engine.path_collector(n_paths)(params)
    if engine.product.error_messages:
        for flag, message in zip(
            np.asarray(jnp.any(values[:, len(engine.payoff_names) :] > 0, axis=0)),
            engine.product.error_messages,
        ):
            if flag:
                raise script_error(message)
    values = values[:, : len(engine.payoff_names)]
    if not np.all(np.isfinite(np.asarray(values))):
        raise InvalidPayoff("non-finite exercise value, feature or payment")
    return values


def _unpack(values, prepared):
    rows = values.reshape(
        (values.shape[0], len(prepared.events), 4 + len(prepared.regression_features))
    )
    return Records(
        rows[:, :, 0], rows[:, :, 1], rows[:, :, 2], rows[:, :, 3:-1], rows[:, :, -1], None
    )


def _advance_targets(values, blocks, event):
    for i, block in enumerate(blocks):
        ratio = (
            block.numeraires[:, event] / block.numeraires[:, event + 1]
            if event + 1 < block.payments.shape[1]
            else 0.0
        )
        values[i] = block.payments[:, event] + ratio * values[i]


def _regression_rows(values, blocks, event, *, scalar=False):
    rows = []
    for value, block in zip(values, blocks):
        x = block.features[:, event, :]
        if scalar and x.shape[-1] == 1:
            x = x[:, 0]
        mask = (block.exercise_values[:, event] > 0.0) & (block.conditions[:, event] > 0.0)
        rows.append((x, value, mask))
    return tuple(rows)


def _backward_fit(prepared, training, validation, degree):
    blocks = [training] + ([] if validation is None else [validation])
    values = [jnp.zeros(block.payments.shape[0], jnp.float64) for block in blocks]
    fits = []
    for event in range(len(prepared.events) - 1, -1, -1):
        _advance_targets(values, blocks, event)
        if exercise_node(prepared.events[event]) is None:
            continue
        rows = _regression_rows(values, blocks, event, scalar=True)
        fit = select_regression(*rows[0], degree, None if len(rows) == 1 else rows[1])
        fits.append(fit)
        for i, (block, row) in enumerate(zip(blocks, rows)):
            exercise = row[2] & (block.exercise_values[:, event] > fit.predict(row[0]))
            values[i] = jnp.where(exercise, block.exercise_values[:, event], values[i])
    return tuple(reversed(fits))


def _backward_device(prepared, training, validation, degree):
    blocks = (training,) if validation is None else (training, validation)
    values = tuple(jnp.zeros(block.payments.shape[0], jnp.float64) for block in blocks)
    features = len(prepared.regression_features)
    width = len(basis_powers(features, degree))
    mask = jnp.asarray(tuple(exercise_node(event) is not None for event in prepared.events))
    ratios = tuple(
        jnp.concatenate(
            (
                block.numeraires[:, :-1] / block.numeraires[:, 1:],
                jnp.zeros((block.payments.shape[0], 1)),
            ),
            axis=1,
        )
        for block in blocks
    )
    empty = (jnp.zeros(width), jnp.zeros(features), jnp.ones(features))

    def step(values, event):
        holding = tuple(
            block.payments[:, event] + ratio[:, event] * value
            for block, ratio, value in zip(blocks, ratios, values, strict=True)
        )

        def exercise(values):
            rows = _regression_rows(values, blocks, event)
            fit = select_device(*rows[0], degree, None if len(rows) == 1 else rows[1])
            updated = tuple(
                jnp.where(
                    row[2] & (block.exercise_values[:, event] > fit.predict(row[0])),
                    block.exercise_values[:, event],
                    value,
                )
                for block, row, value in zip(blocks, rows, values, strict=True)
            )
            return updated, (fit.coefficients, fit.means, fit.sigmas)

        return jax.lax.cond(mask[event], exercise, lambda values: (values, empty), holding)

    _, fields = jax.lax.scan(step, values, jnp.arange(len(prepared.events)), reverse=True)
    return Policy(*fields, mask)


def _fuzzy_value(prepared, records, policy, ctx):
    mask = policy.exercise_mask
    eps = jnp.asarray(
        tuple(
            ctx.smooth if (node := exercise_node(event)) is None or node.eps < 0 else node.eps
            for event in prepared.events
        ),
        dtype=records.payments.dtype,
    )
    kernels = (
        (cspr, cspr_bounds)
        if ctx.smoothing_kernel == "dal"
        else (smoothstep_cspr, smoothstep_cspr_bounds)
    )
    ratios = jnp.concatenate(
        (records.numeraires[:-1] / records.numeraires[1:], jnp.zeros(1, records.numeraires.dtype))
    )
    continuation = jax.vmap(policy.predict)(jnp.arange(len(prepared.events)), records.features)
    degrees = (
        kernels[0](records.exercise_values - continuation, eps)
        * kernels[1](records.exercise_values, 0.0, eps)
        * records.conditions
    )
    degrees = jnp.where(mask, degrees, 0.0)

    def step(value, row):
        payment, h, degree, ratio = row
        holding = payment + ratio * value
        return degree * h + (1.0 - degree) * holding, None

    initial = 0.0 * records.numeraires[0]
    body = jax.checkpoint(step, prevent_cse=False) if ctx.checkpoint else step
    value, _ = jax.lax.scan(
        body,
        initial,
        (records.payments, records.exercise_values, degrees, ratios),
        reverse=True,
        unroll=4,
    )
    return value / records.numeraires[0]


class _ReplayEngine(MonteCarloEngine):
    """Dynamic replica shift/offset, so one compiled price vmaps over keys."""

    def _random_normals(self, params, block_id, path_ids):
        if "_lsmc_random" not in params:
            return super()._random_normals(params, block_id, path_ids)
        offset, shift = params["_lsmc_random"]
        sequence = Sobol(dim=self.sim_dim)
        states = jax.vmap(sequence.state)(path_ids + offset)
        uniforms = ((states ^ shift).astype(jnp.float64) + 0.5) * MUL
        normals = inverse_ncdf(
            uniforms,
            self.settings.inverse_normal == "acklam_polish_precise",
            self.settings.inverse_normal in ("acklam_polish", "acklam_polish_precise"),
        )
        if self.settings.inverse_normal == "ndtri":
            normals = inverse_ncdf_ndtri(uniforms)
        if self.settings.use_bb:
            plan = bridge.bridge_plan(self.sim_dim // self.model.n_factors)
            normals = jax.vmap(
                lambda z: bridge.apply_factors(plan, z, n_factors=self.model.n_factors)
            )(normals)
        return normals


@dataclass(frozen=True, init=False, eq=False)
class LsmcEngine:
    """Host training and a pure, differentiable frozen-policy pricing function."""

    prepared: "PreparedProduct"
    model: Model
    settings: MonteCarloSettings
    _training: MonteCarloEngine
    _validation_engines: dict
    _replica_functions: dict
    _value_functions: dict
    _policy_risk_functions: dict

    def __init__(self, prepared, model, settings=None):
        object.__setattr__(self, "prepared", prepared)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "settings", settings or MonteCarloSettings())
        if self.settings.rsg != "sobol":
            raise script_error("UnsupportedRsgForExercise: EXERCISE requires rsg=sobol")
        if len(prepared.regression_features) > 1 and self.settings.lsmc_basis_degree > 3:
            raise script_error("InvalidLsmcFeatureBudget: multivariate basis degree must be 1..3")
        training = MonteCarloEngine(
            _records_product(prepared, 0),
            model,
            replace(
                self.settings,
                enable_aad=False,
                sobol_shift_key=self.settings.sobol_shift_key
                if self.settings.lsmc_rqmc_replicates is None
                else scramble_key(False, self.settings.lsmc_training_seed or 0),
                lsmc_policy_risk_mode="Frozen",
            ),
        )
        object.__setattr__(self, "_training", training)
        for name in (
            "_validation_engines",
            "_replica_functions",
            "_value_functions",
            "_policy_risk_functions",
        ):
            object.__setattr__(self, name, {})

    def default_params(self):
        return self._training.default_params()

    @property
    def devices(self):
        return self._training.devices

    @property
    def dtype(self):
        return self._training.dtype

    def layout(self, n_paths):
        _path_counts(self.settings, n_paths)
        return self._training.layout(n_paths)

    def _validation_engine(self, n_train):
        if n_train not in self._validation_engines:
            self._validation_engines[n_train] = MonteCarloEngine(
                _records_product(self.prepared, n_train), self.model, self._training.settings
            )
        return self._validation_engines[n_train]

    def train_result(self, n_paths, params=None) -> TrainingResult:
        params = self.default_params() if params is None else params
        self.model.validate_params(params["model"])
        n_train, n_val, _ = _path_counts(self.settings, n_paths)
        train = _unpack(_collect(self._training, params, n_train), self.prepared)
        validation = None
        if n_val:
            validation_engine = self._validation_engine(n_train)
            validation = _unpack(_collect(validation_engine, params, n_val), self.prepared)
        regressions = _backward_fit(
            self.prepared, train, validation, self.settings.lsmc_basis_degree
        )
        return TrainingResult(
            _policy(regressions, self.prepared, self.settings.lsmc_basis_degree), regressions
        )

    def train(self, n_paths, params=None) -> Policy:
        return self.train_result(n_paths, params).policy

    def train_jax(self, n_paths, params):
        """Host training wrapper; construct training_pricer before applying transforms."""
        return self.training_pricer(n_paths)(params)

    def training_pricer(self, n_paths):
        """Build a pure fixed-shape training function before applying JAX transforms."""
        n_train, n_val, _ = _path_counts(self.settings, n_paths)
        width = len(self._training.payoff_names)
        collect_train = self._training.path_collector(n_train)
        collect_validation = (
            self._validation_engine(n_train).path_collector(n_val) if n_val else None
        )

        def fit(params):
            train = _unpack(collect_train(params)[:, :width], self.prepared)
            validation = (
                None
                if collect_validation is None
                else _unpack(collect_validation(params)[:, :width], self.prepared)
            )
            return _backward_device(
                self.prepared, train, validation, self.settings.lsmc_basis_degree
            )

        return fit

    def _replay(self, n_paths, policy):
        n_train, n_val, _ = _path_counts(self.settings, n_paths)
        errors = self._training.product.error_messages

        def payoff(params, scenario, ctx, initial):
            selected = params["_lsmc_policy"] if policy is None else policy
            recorder = lower_records(self.prepared, ctx, policy=None if ctx.fuzzy else selected)
            records, hard = recorder(
                initial, regression_scenario(self.prepared, scenario), params["script"]
            )
            value = _fuzzy_value(self.prepared, records, selected, ctx) if ctx.fuzzy else hard
            return jnp.concatenate((value[None], records.errors.astype(value.dtype)))

        product = PathProduct(
            timeline=self.prepared.timeline,
            sample_defs=self.prepared.sample_defs,
            payoff=payoff,
            script_params=self.prepared.script_params,
            initial_state=self._training.product.initial_state,
            error_messages=errors,
            path_offset=n_train + n_val,
            path_state_size=self._training.product.path_state_size,
        )
        return _ReplayEngine(product, self.model, self.settings)

    def replica_pricer(self, n_paths, policy, *, checked=False):
        """Pure ``params -> prices[replicas]``; policy is explicit and frozen."""
        frozen = jax.tree.map(jax.lax.stop_gradient, policy)
        key = n_paths, checked
        if key not in self._replica_functions:
            self._replica_functions[key] = self._raw_replica_pricer(n_paths, checked)
        fn = self._replica_functions[key]
        return lambda params: fn(dict(params, _lsmc_policy=frozen))

    def _raw_replica_pricer(self, n_paths, checked):
        replay = self._replay(n_paths, None)
        price = replay.checked_pricer(n_paths) if checked else replay.pricer(n_paths)
        if self.settings.lsmc_rqmc_replicates is None:
            if checked:

                def single(params):
                    prices, errors = price(params)
                    return prices.reshape(1), errors[None, :]

                return single
            return lambda params: price(params).reshape(1)
        shifts = jnp.asarray(
            np.stack(
                tuple(
                    digital_shifts(
                        replay.sim_dim, scramble_key(True, self.settings.lsmc_pricing_seed or 0, r)
                    )
                    for r in range(self.settings.lsmc_rqmc_replicates)
                )
            )
        )
        offsets = jnp.arange(self.settings.lsmc_rqmc_replicates, dtype=jnp.int64) * n_paths

        def replicas(params):
            output = jax.vmap(
                lambda offset, shift: price(dict(params, _lsmc_random=(offset, shift)))
            )(offsets, shifts)
            return (output[0][:, 0], output[1]) if checked else output[:, 0]

        return replicas

    def pricer(self, n_paths, *, policy=None, training_params=None):
        """Train now (if needed), then return a JAX function for the frozen policy."""
        policy = self.train(n_paths, training_params) if policy is None else policy
        prices = self.replica_pricer(n_paths, policy)
        return lambda params: jnp.mean(prices(params))[None]

    def value(self, n_paths, params=None):
        return self.evaluate(n_paths, params).as_dict()

    def evaluate(self, n_paths, params=None) -> LsmcResult:
        params = self.default_params() if params is None else params
        training = self.train_result(n_paths, params)
        policy = training.policy
        has_errors = bool(self._training.product.error_messages)
        self.replica_pricer(n_paths, policy, checked=has_errors)
        replicas = self._replica_functions[n_paths, has_errors]

        # Policy arrays are dynamic arguments so repeated training reuses the executable.
        def price(p, selected):
            frozen = jax.tree.map(jax.lax.stop_gradient, selected)
            replica_price = lambda p: replicas(dict(p, _lsmc_policy=frozen))
            if has_errors:
                prices, errors = replica_price(p)
            else:
                prices, errors = replica_price(p), jnp.zeros((1, 1))
            return jnp.mean(prices), (prices, errors)

        if n_paths not in self._value_functions:
            self._value_functions[n_paths] = jax.jit(
                jax.value_and_grad(price, has_aux=True) if self.settings.enable_aad else price
            )
        compiled = self._value_functions[n_paths]
        if self.settings.enable_aad:
            (pv, (prices, errors)), grads = compiled(params, policy)
            result = self._risk_results(pv, grads)
        else:
            pv, (prices, errors) = compiled(params, policy)
            result = {"PV": float(pv)}
        replicate_means = tuple(float(value) for value in prices)
        self._validate_result(result, errors)
        if self.settings.lsmc_policy_risk_mode == "RetrainedBump":
            correction = self.policy_risk_correction(n_paths, params, policy)
            self._add_policy_risk(result, correction)
        return LsmcResult(tuple(result.items()), replicate_means, training)

    def clear_cache(self):
        self._training.clear_cache()
        for engine in self._validation_engines.values():
            engine.clear_cache()
        for cache in (
            self._validation_engines,
            self._replica_functions,
            self._value_functions,
            self._policy_risk_functions,
        ):
            cache.clear()

    def _risk_results(self, pv, grads):
        result = {"PV": float(pv)} | {
            f"d_{label}": float(grads["model"][label]) for label in self.model.param_labels
        }
        return result | {
            f"d_{name}": float(grads["script"][name]) for name, _ in self.prepared.script_params
        }

    def _validate_result(self, result, errors):
        for flag, message in zip(
            np.asarray(jnp.any(errors > 0, axis=0)), self._training.product.error_messages
        ):
            if flag:
                raise script_error(message)
        if not all(np.isfinite(v) for v in result.values()):
            raise InvalidPayoff("non-finite LSMC price or risk")

    @staticmethod
    def _add_policy_risk(result, correction):
        for group in ("model", "script"):
            for name, value in correction[group].items():
                result[f"d_{name}"] += float(value)

    def policy_risk_correction(self, n_paths, params, base_policy):
        """Reprice bumped policies at base inputs; all retraining uses vmap."""
        from jax.flatten_util import ravel_pytree

        flat, unravel = ravel_pytree(params)
        steps = self.settings.lsmc_policy_bump_relative * jnp.maximum(1.0, jnp.abs(flat))
        if not np.isfinite(np.asarray(steps)).all() or np.any(np.asarray(steps) <= 0.0):
            raise script_error("InvalidLsmcPolicyBump: parameter bumps must be finite and positive")
        bumps = np.asarray(jnp.eye(flat.shape[0]) * steps)
        upper, lower = np.asarray(flat) + bumps, np.asarray(flat) - bumps
        lower_valid = []
        for up, down in zip(upper, lower):
            if not np.isfinite(up).all():
                raise script_error("InvalidLsmcPolicyBump: upper bumped parameters must be finite")
            try:
                self.model.validate_params(unravel(jnp.asarray(up))["model"])
            except (DalError, ValueError, OverflowError) as error:
                raise script_error(
                    "InvalidLsmcPolicyBump: upper model parameter violates its constraint"
                ) from error
            try:
                self.model.validate_params(unravel(jnp.asarray(down))["model"])
                lower_valid.append(bool(np.isfinite(down).all()))
            except (DalError, ValueError, OverflowError):
                lower_valid.append(False)
        lower_valid = jnp.asarray(lower_valid)
        lower = jnp.where(lower_valid[:, None], jnp.asarray(lower), flat[None, :])
        inputs = jnp.concatenate((jnp.asarray(upper), lower))
        if n_paths not in self._policy_risk_functions:
            fit = self.training_pricer(n_paths)
            replicas = self._raw_replica_pricer(n_paths, False)

            def bumped_prices(rows, base_params):
                def value(row):
                    bumped = jax.tree.map(jax.lax.stop_gradient, unravel(row))
                    policy = fit(bumped)
                    return jnp.mean(replicas(dict(base_params, _lsmc_policy=policy)))

                return jax.vmap(value)(rows)

            def base_price(base_params, policy):
                return jnp.mean(replicas(dict(base_params, _lsmc_policy=policy)))

            self._policy_risk_functions[n_paths] = jax.jit(bumped_prices), jax.jit(base_price)
        bumped_prices, base_price = self._policy_risk_functions[n_paths]
        values = bumped_prices(inputs, params)
        size = flat.shape[0]
        base = base_price(params, base_policy)
        down = jnp.where(lower_valid, values[size:], base)
        correction = (values[:size] - down) / jnp.where(lower_valid, 2.0 * steps, steps)
        return unravel(correction)
