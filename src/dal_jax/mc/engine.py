"""Block-based Monte Carlo engine."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike

from dal_jax.errors import (
    InvalidPathCount,
    InvalidPayoff,
    InvalidSetting,
    ReservedIdentifier,
    UnsupportedBrownianBridge,
    script_error,
)
from dal_jax.mc import parallel, tuning
from dal_jax.mc.settings import DEFAULT_SMOOTH, MonteCarloSettings
from dal_jax.models.base import Model, SampleDef
from dal_jax.random import bridge
from dal_jax.random.inverse_normal import inverse_ncdf_ndtri
from dal_jax.random.prng import block_normals, prng_key
from dal_jax.random.sobol import MAX_POINTS, Sobol

type Params = Mapping[str, Mapping[str, ArrayLike]]


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalContext:
    """How a payoff is evaluated: exact (hard conditions) or fuzzy (smoothed, for gradients)."""

    fuzzy: bool = False
    smooth: float = DEFAULT_SMOOTH
    smoothing_kernel: str = "dal"
    scan_group_threshold: int = 4
    axis_name: str | None = None
    checkpoint: bool = True


type PathPayoff = Callable[..., ArrayLike]


@dataclass(frozen=True, slots=True, kw_only=True)
class PathProduct:
    """Timeline and single-path payoff; initial_state adds a shared fourth payoff argument."""

    timeline: tuple[float, ...]
    payoff: PathPayoff
    payoff_names: tuple[str, ...] = ("PV",)
    sample_defs: tuple[SampleDef, ...] | None = None
    script_params: tuple[tuple[str, float], ...] | Mapping[str, float] = field(default=())
    initial_state: Callable[[Params], ArrayLike] | None = None
    error_messages: tuple[str, ...] = ()
    path_state_size: int = 0
    path_offset: int = 0

    def __post_init__(self) -> None:
        timeline = tuple(float(t) for t in self.timeline)
        object.__setattr__(self, "timeline", timeline)
        defs = (
            tuple(SampleDef() for _ in timeline)
            if self.sample_defs is None
            else tuple(self.sample_defs)
        )
        object.__setattr__(self, "sample_defs", defs)
        object.__setattr__(self, "script_params", _script_params(self.script_params))
        object.__setattr__(self, "error_messages", tuple(self.error_messages))
        names = tuple(self.payoff_names)
        if not names or len(set(names)) != len(names):
            raise InvalidSetting("payoff_names must be non-empty and unique")
        object.__setattr__(self, "payoff_names", names)
        _nonnegative_size("path_state_size", self.path_state_size)
        _nonnegative_size("path_offset", self.path_offset)


def _nonnegative_size(name, value):
    if not isinstance(value, int) or value < 0:
        raise InvalidSetting(f"{name} must be a nonnegative integer")


def _script_params(
    params: tuple[tuple[str, float], ...] | Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    items = params.items() if isinstance(params, Mapping) else params
    normalized = tuple((str(name), float(value)) for name, value in items)
    if len({name for name, _ in normalized}) != len(normalized):
        raise InvalidSetting("script parameter names must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class BlockLayout:
    block_size: int
    n_blocks: int  # padded to a multiple of n_devices (and to a power of two when bucketing)
    n_devices: int


def _check_risk_labels(product: PathProduct, model: Model) -> None:
    """Model and script parameters share the ``d_<name>`` namespace of the results."""
    clashes = sorted(set(model.param_labels) & {name for name, _ in product.script_params})
    if clashes:
        raise ReservedIdentifier(
            f"script parameters {clashes} reuse model parameter labels; both would report as d_<name>"
        )


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True, init=False, eq=False)
class MonteCarloEngine:
    product: PathProduct
    model: Model
    settings: MonteCarloSettings
    expired: bool
    plan: object
    sim_dim: int
    devices: tuple[jax.Device, ...]
    dtype: np.dtype
    block_size: int
    _replicated: NamedSharding
    _normals: Callable
    _compiled: dict[tuple, Callable]
    _record_functions: dict[int, Callable]

    def __init__(
        self, product: PathProduct, model: Model, settings: MonteCarloSettings | None = None
    ) -> None:
        settings = settings or MonteCarloSettings()
        _check_risk_labels(product, model)
        expired = not product.timeline
        plan = None if expired else model.allocate(product.timeline, product.sample_defs)
        sim_dim = 0 if expired else model.sim_dim(plan)
        if settings.use_bb and not model.supports_bb:
            raise UnsupportedBrownianBridge("model does not support a factor-aware bridge")
        devices = settings.resolved_devices()
        devices = devices[:1] if settings.parallel == "none" else devices
        dtype = jnp.dtype(tuning.resolve_dtype(settings.dtype, devices))
        values = dict(
            product=product,
            model=model,
            settings=settings,
            expired=expired,
            plan=plan,
            sim_dim=sim_dim,
            devices=devices,
            dtype=dtype,
            block_size=tuning.resolve_block_size(settings, product, sim_dim, devices, dtype),
            _replicated=NamedSharding(parallel.make_mesh(devices), P()),
            _compiled={},
            _record_functions={},
        )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_normals", self._make_normals())

    @property
    def payoff_names(self) -> tuple[str, ...]:
        return self.product.payoff_names

    @property
    def _output_count(self) -> int:
        return len(self.payoff_names) + len(self.product.error_messages)

    @property
    def script_param_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.product.script_params)

    def default_params(self) -> dict[str, dict[str, Array]]:
        return jax.device_put(
            {
                "model": self.model.default_params(),
                "script": {
                    name: jnp.asarray(value, dtype=jnp.float64)
                    for name, value in self.product.script_params
                },
            },
            self._replicated,
        )

    def layout(self, n_paths: int) -> BlockLayout:
        if not isinstance(n_paths, (int, np.integer)) or n_paths <= 0:
            raise InvalidPathCount("number of paths must be positive")
        n_paths = int(n_paths)
        block_size = (
            self.block_size if self.settings.block_bucketing else min(self.block_size, n_paths)
        )
        n_blocks = math.ceil(n_paths / block_size)
        if self.settings.block_bucketing:
            n_blocks = _next_pow2(n_blocks)
        n_devices = len(self.devices)
        n_blocks = math.ceil(n_blocks / n_devices) * n_devices
        if (
            self.settings.rsg == "sobol"
            and self.product.path_offset + n_blocks * block_size > MAX_POINTS
        ):
            raise InvalidPathCount(
                f"Sobol supports at most {MAX_POINTS} paths including block padding"
            )
        return BlockLayout(block_size, n_blocks, n_devices)

    def _make_normals(self) -> Callable[[Array, Array], Array]:
        """``(block_id, path_ids[B]) -> float64[B, sim_dim]``, bridged if ``use_bb``."""
        settings, dim = self.settings, self.sim_dim
        if dim == 0:
            return lambda block_id, path_ids: jnp.zeros((path_ids.shape[0], 0), dtype=jnp.float64)

        if settings.rsg == "sobol":
            sequence = Sobol(
                dim=dim,
                shift_key=settings.sobol_shift_key,
                precise=settings.inverse_normal == "acklam_polish_precise",
                polish=settings.inverse_normal in ("acklam_polish", "acklam_polish_precise"),
            )
            if settings.inverse_normal == "ndtri":
                one_path = lambda path_id: inverse_ncdf_ndtri(sequence.uniform(path_id))
            else:
                one_path = sequence.normal
            draw = lambda block_id, path_ids: jax.vmap(one_path)(path_ids)
        else:
            draw = lambda block_id, path_ids: block_normals(
                prng_key(settings.rsg, settings.seed, settings.prng_impl),
                block_id,
                path_ids.shape[0],
                dim,
            )

        if not settings.use_bb:
            return draw
        n_factors = self.model.n_factors
        plan = bridge.bridge_plan(dim // n_factors)
        one_bridge = partial(bridge.apply_factors, plan, n_factors=n_factors)
        return lambda block_id, path_ids: jax.vmap(one_bridge)(draw(block_id, path_ids))

    def _cast(self, tree):
        """Float leaves to the path dtype (a no-op in the default float64 mode)."""
        if self.dtype == jnp.float64:
            return tree
        return jax.tree.map(
            lambda x: (
                jnp.asarray(x, self.dtype)
                if jnp.issubdtype(jnp.result_type(x), jnp.floating)
                else x
            ),
            tree,
        )

    def _model_state(self, params: Params):
        return self._cast(self.model.init(params["model"], self.plan))

    def _random_normals(self, params, block_id, path_ids):
        return self._normals(block_id, path_ids)

    def _simulation_state(self, params: Params):
        """Model precomputations and optional product history, outside path/block loops."""
        initial = (
            None
            if self.product.initial_state is None
            else self._cast(self.product.initial_state(params))
        )
        return self._model_state(params), initial

    def _block_paths(
        self, params: Params, state, block_id: Array, block_size: int, ctx: EvalContext
    ) -> tuple[Array, Array]:
        path_ids = (
            self.product.path_offset
            + block_id * block_size
            + jnp.arange(block_size, dtype=jnp.int64)
        )
        normals = self._random_normals(params, block_id, path_ids).astype(self.dtype)
        cast_params = self._cast(params)
        n_payoffs = self._output_count
        model_state, initial = state

        def one_path(z):
            scenario = self.model.generate(model_state, self.plan, z)
            args = () if self.product.initial_state is None else (initial,)
            return jnp.reshape(
                jnp.asarray(self.product.payoff(cast_params, scenario, ctx, *args)), (n_payoffs,)
            )

        return path_ids, jax.vmap(one_path)(normals)

    def _block_sum(
        self,
        params: Params,
        state,
        block_id: Array,
        n_paths: Array,
        block_size: int,
        ctx: EvalContext,
    ) -> Array:
        path_ids, values = self._block_paths(params, state, block_id, block_size, ctx)
        live = (path_ids < self.product.path_offset + n_paths)[:, None]
        return jnp.sum(jnp.where(live, values, 0.0), axis=0).astype(jnp.float64)

    def _context(self, fuzzy: bool | None) -> EvalContext:
        return EvalContext(
            fuzzy=self.settings.enable_aad if fuzzy is None else fuzzy,
            smooth=self.settings.smooth,
            smoothing_kernel=self.settings.smoothing_kernel,
            scan_group_threshold=self.settings.scan_group_threshold,
            checkpoint=self.settings.checkpoint,
        )

    def _core(self, layout: BlockLayout, ctx: EvalContext) -> Callable[[Params, Array], Array]:
        """``(params, n_paths) -> pv[n_payoffs]``; ``n_paths`` is traced so bucketed layouts share a compilation."""
        n_payoffs = self._output_count
        if self.expired:
            return lambda params, n_paths: jnp.zeros(n_payoffs, dtype=jnp.float64)

        block_sum = partial(self._block_sum, block_size=layout.block_size)
        strategy = dict(
            strategy=self.settings.parallel, devices=self.devices, n_blocks=layout.n_blocks
        )

        if not self.settings.deterministic_reduction:

            def local(params, block_ids, n_paths, axis_name):
                body = partial(block_sum, ctx=replace(ctx, axis_name=axis_name))
                body_fn = (
                    jax.checkpoint(body, prevent_cse=False) if self.settings.checkpoint else body
                )
                state = self._simulation_state(params)
                acc = jnp.zeros(n_payoffs, dtype=jnp.float64)
                if axis_name is not None:
                    acc = jax.lax.pcast(acc, (axis_name,), to="varying")
                acc, _ = jax.lax.scan(
                    lambda a, b: (a + body_fn(params, state, b, n_paths), None), acc, block_ids
                )
                return acc

            return lambda params, n_paths: (
                parallel.sum_blocks(local, params, n_paths, **strategy) / n_paths
            )

        #  Deterministic reduction: per-block values (and Jacobians when differentiating)
        #  are gathered in block order and summed sequentially, so neither the device
        #  count nor the psum tree changes a single bit.  Reverse mode only.
        def block_value(params, block_id, n_paths, axis_name):
            return block_sum(
                params,
                self._simulation_state(params),
                block_id,
                n_paths,
                ctx=replace(ctx, axis_name=axis_name),
            )

        def values_local(params, block_ids, n_paths, axis_name):
            return jax.lax.map(lambda b: block_value(params, b, n_paths, axis_name), block_ids)

        def jacobians_local(params, block_ids, n_paths, axis_name):
            basis = jnp.eye(n_payoffs, dtype=jnp.float64)
            if axis_name is not None:
                basis = jax.lax.pcast(basis, (axis_name,), to="varying")

            def value_and_jacobian(block_id):
                value, vjp = jax.vjp(lambda p: block_value(p, block_id, n_paths, axis_name), params)
                (jacobian,) = jax.vmap(vjp)(basis)
                return value, jacobian

            return jax.lax.map(value_and_jacobian, block_ids)

        @jax.custom_vjp
        def pv(params, n_paths):
            return (
                parallel.ordered_sum(
                    parallel.gather_blocks(values_local, params, n_paths, **strategy)
                )
                / n_paths
            )

        def pv_fwd(params, n_paths):
            values, jacobians = parallel.gather_blocks(jacobians_local, params, n_paths, **strategy)
            return parallel.ordered_sum(values) / n_paths, (
                jax.tree.map(parallel.ordered_sum, jacobians),
                n_paths,
            )

        def pv_bwd(residuals, cotangent):
            jacobians, n_paths = residuals
            return jax.tree.map(
                lambda j: jnp.tensordot(cotangent, j, axes=1) / n_paths, jacobians
            ), None

        pv.defvjp(pv_fwd, pv_bwd)
        return pv

    def pricer(self, n_paths: int, *, fuzzy: bool | None = None) -> Callable[[Params], Array]:
        """Pure ``f(params) -> pv[n_payoffs]`` over ``n_paths`` paths.

        ``fuzzy`` defaults to ``settings.enable_aad`` (DAL prices and
        differentiates in fuzzy mode when AAD is enabled).
        """
        core = self._core(self.layout(n_paths), self._context(fuzzy))
        if not self.product.error_messages:
            return lambda params: core(params, jnp.asarray(n_paths, dtype=jnp.int64))

        def price(params):
            result = core(params, jnp.asarray(n_paths, dtype=jnp.int64))
            return jnp.where(
                jnp.any(result[len(self.payoff_names) :] > 0),
                jnp.nan,
                result[: len(self.payoff_names)],
            )

        return price

    def checked_pricer(self, n_paths: int, *, fuzzy: bool | None = None):
        """Pure ``params -> (prices, error_rates)`` for host-side named error checks."""
        core = self._core(self.layout(n_paths), self._context(fuzzy))

        def price(params):
            values = core(params, jnp.asarray(n_paths, dtype=jnp.int64))
            return values[: len(self.payoff_names)], values[len(self.payoff_names) :]

        return price

    def path_payoffs(
        self, params: Params, block_id: ArrayLike, n_paths: int, *, fuzzy: bool | None = None
    ) -> tuple[Array, Array]:
        """``(path_ids[B], values[B, n_payoffs])`` for one block of the ``n_paths`` layout.

        Pure and jittable; ids at or beyond ``n_paths`` are block padding.
        """
        if self.expired:
            raise InvalidSetting("an expired product has no paths")
        layout = self.layout(n_paths)
        block_id = jax.device_put(jnp.asarray(block_id, dtype=jnp.int64), self._replicated)
        ids, values = self._block_paths(
            params,
            self._simulation_state(params),
            block_id,
            layout.block_size,
            self._context(fuzzy),
        )
        prices = values[:, : len(self.payoff_names)]
        if self.product.error_messages:
            prices = jnp.where(
                jnp.any(values[:, len(self.payoff_names) :] > 0, axis=1)[:, None], jnp.nan, prices
            )
        return ids, prices

    def path_collector(self, n_paths: int) -> Callable[[Params], Array]:
        """Collect exact path outputs and error flags in global path order."""
        layout = self.layout(n_paths)
        if n_paths not in self._record_functions:

            def local(params, block_ids, count, axis_name):
                state = self._simulation_state(params)
                context = replace(self._context(False), axis_name=axis_name)
                return jax.lax.map(
                    lambda block: self._block_paths(
                        params, state, block, layout.block_size, context
                    )[1],
                    block_ids,
                )

            def collect(params):
                values = parallel.gather_blocks(
                    local,
                    params,
                    jnp.asarray(n_paths),
                    strategy=self.settings.parallel,
                    devices=self.devices,
                    n_blocks=layout.n_blocks,
                )
                return values.reshape((-1, self._output_count))[:n_paths]

            self._record_functions[n_paths] = jax.jit(collect)
        return self._record_functions[n_paths]

    def clear_cache(self) -> None:
        """Release this engine's references to compiled pricing and collection functions."""
        self._compiled.clear()
        self._record_functions.clear()

    def value(
        self, n_paths: int, params: Params | None = None, *, payoff: str | None = None
    ) -> dict[str, float | np.ndarray]:
        """DAL-style result ``{"PV": ..., "d_<label>": ...}`` for one payoff.

        With ``enable_aad`` the price and all ``d_<label>`` come from the fuzzy
        evaluation via ``jax.value_and_grad``; otherwise only the exact ``PV``.
        """
        params = self.default_params() if params is None else params
        self.model.validate_params(params["model"])
        index = 0 if payoff is None else self.payoff_names.index(payoff)
        params = jax.device_put(params, self._replicated)
        compiled = self._value_function(self.layout(n_paths), index)
        count = jax.device_put(jnp.asarray(n_paths, dtype=jnp.int64), self._replicated)
        output = compiled(params, count)
        if self.product.error_messages:
            output, rates = output
            for rate, message in zip(np.asarray(rates), self.product.error_messages):
                if rate > 0:
                    raise script_error(message)
        result = self._dal_result(output)
        if not all(np.all(np.isfinite(v)) for v in result.values()):
            raise InvalidPayoff("non-finite path value")
        return result

    def _value_function(self, layout: BlockLayout, index: int) -> Callable:
        """Jitted ``(params, n_paths) -> pv`` (or ``(pv, grads)`` with AAD), cached per layout and payoff."""
        key = (layout, self.settings.enable_aad, index)
        if key not in self._compiled:
            core = self._core(layout, self._context(None))
            self._compiled[key] = jax.jit(self._host_function(core, index))
        return self._compiled[key]

    def _host_function(self, core, index):
        if not self.product.error_messages:
            scalar = lambda p, n: core(p, n)[index]
            return jax.value_and_grad(scalar) if self.settings.enable_aad else scalar

        def checked_scalar(p, n):
            result = core(p, n)
            return result[index], result[len(self.payoff_names) :]

        if not self.settings.enable_aad:
            return checked_scalar
        derivative = jax.value_and_grad(checked_scalar, has_aux=True)

        def value(p, n):
            (pv, errors), grads = derivative(p, n)
            return (pv, grads), errors

        return value

    def _dal_result(self, out) -> dict[str, float | np.ndarray]:
        if not self.settings.enable_aad:
            return {"PV": _to_host(out)}
        pv, grads = out
        result: dict[str, float | np.ndarray] = {"PV": _to_host(pv)}
        result |= {
            f"d_{label}": _to_host(grads["model"][label]) for label in self.model.param_labels
        }
        result |= {f"d_{name}": _to_host(grads["script"][name]) for name in self.script_param_names}
        return result


def _to_host(x: Array) -> float | np.ndarray:
    host = np.asarray(x)
    return float(host) if host.ndim == 0 else host
