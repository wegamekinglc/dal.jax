"""DAL-compatible preparation and hard-policy simulation diagnostics."""

import json
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from dal_jax.errors import InvalidPayoff, script_error
from dal_jax.mc.lsmc import _path_counts, _ReplayEngine, scramble_key
from dal_jax.mc.regression import basis_powers
from dal_jax.mc.settings import MonteCarloSettings
from dal_jax.random.sobol import digital_shifts
from dal_jax.script.diagnostics import describe
from dal_jax.script.fixings import ValuationSettings
from dal_jax.script.lower.lsmc import lower_records, regression_scenario
from dal_jax.script.lsmcprep import exercise_node
from dal_jax.script.preparation import prepare
from dal_jax.strings import ci_key


def simulation_settings(settings):
    names = (
        "use_bb",
        "enable_aad",
        "smooth",
        "lsmc_basis_degree",
        "lsmc_training_paths",
        "lsmc_validation_paths",
        "lsmc_rqmc_replicates",
        "lsmc_training_seed",
        "lsmc_pricing_seed",
    )
    return {
        "rsg": settings.rsg,
        **{name: getattr(settings, name) for name in names[:3]},
        "compiled": False,
        **{name: getattr(settings, name) for name in names[3:]},
    }


def _nodes(node):
    if isinstance(node, dict):
        if "kind" in node:
            yield node
        for value in node.values():
            yield from _nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _nodes(value)


def _observation_uses(description):
    uses = {}
    for event in description["events"]:
        for statement_id, statement in enumerate(event["statements"]):
            for node in _nodes(statement):
                if node["kind"] not in ("fix", "spot") or not node.get("index_canonical"):
                    continue
                day = node["fixing_date_literal"] or event["date"]
                key = ci_key(node["index_canonical"]), day
                use = {
                    "event_id": event["event_id"],
                    "statement_id": statement_id,
                    "node_id": node["id"],
                    "source": node["source"],
                    "index_original": node["index_original"],
                    "fixing_date_mode": node["fixing_date_mode"],
                    "fixing_date_literal": node["fixing_date_literal"],
                    "observation_type": "Spot" if node["kind"] == "spot" else "Fix",
                }
                uses.setdefault(key, []).append(use)
    return uses


def _model_bindings(prepared):
    names = [
        record.index_name
        for record in prepared.observations
        if record.index_name and not record.historical
    ]
    names.extend(
        feature.name
        for feature in prepared.regression_features
        if feature.variable_index is None and feature.name != "SPOT"
    )
    unique = {}
    for name in names:
        unique.setdefault(ci_key(name), name)
    return tuple(unique.values())


def _observation_request(record, request_id, uses, history_id, expired):
    request = {
        "request_id": request_id,
        "index_canonical": record.index_name,
        "fixing_time": f"{record.date} 00:00:00",
        "source": "Historical" if record.historical else "Model",
        "resolution": "SkippedExpired" if expired else "Resolved",
        "uses": uses.get((ci_key(record.index_name), str(record.date)), []),
        "history_value_id": None,
        "value": None,
        "model_slot": None,
    }
    if record.historical and not expired:
        request.update(history_value_id=history_id, value=record.value)
    elif record.sample_id is not None and record.output_id is not None:
        request["model_slot"] = {"sample_id": record.sample_id, "output_id": record.output_id}
    return request


def _observation_requests(prepared, description):
    uses = _observation_uses(description)
    requests = []
    history_id = 0
    for record in prepared.observations:
        if not record.index_name:
            continue
        request = _observation_request(record, len(requests), uses, history_id, prepared.expired)
        history_id += int(request["history_value_id"] is not None)
        requests.append(request)
    return requests


def _valuation_topology(prepared, description):
    all_dates = [event["date"] for event in description["events"]]
    live = [
        {"event_id": all_dates.index(str(date)), "future_event_index": i}
        for i, date in enumerate(prepared.event_dates)
    ]
    definitions = [
        {
            "sample_id": i,
            "numeraire": definition.numeraire,
            "index_names": list(definition.index_names),
            "discount_maturities": list(definition.discount_mats),
            "forward_maturities": [],
            "libor_definitions": [],
        }
        for i, definition in enumerate(prepared.sample_defs)
    ]
    return {
        "sample_dates": [
            str(prepared.evaluation_date.add_days(round(time * 365.0)))
            for time in prepared.timeline
        ],
        "timeline": list(prepared.timeline),
        "event_to_sample": list(prepared.event_to_sample),
        "live_events": live,
        "sample_definitions": definitions,
        "numeraire_requests": [
            dict(event, sample_id=sample) for event, sample in zip(live, prepared.event_to_sample)
        ],
    }


def valuation_explain(data, model, valuation=None):
    valuation = valuation or ValuationSettings()
    prepared = prepare(data, model=model, valuation=valuation)
    description = json.loads(describe(data))
    requests = _observation_requests(prepared, description)
    bindings = _model_bindings(prepared)
    return _valuation_topology(prepared, description) | {
        "schema": "dal.script-valuation/1",
        "evaluation_date": str(prepared.evaluation_date),
        "today_fixing": str(valuation.today_fixing_policy),
        "source_kind": "GlobalSnapshot" if valuation.fixings is None else "ExplicitSnapshot",
        "simulation": simulation_settings(MonteCarloSettings()),
        "all_expired": prepared.expired,
        "observation_mode": "Named" if requests else "Legacy",
        "model_bindings": [
            {
                "asset": "spot" if len(bindings) == 1 else name,
                "index_original": name,
                "index_canonical": name,
            }
            for name in bindings
        ],
        "requests": requests,
    }


def _replay_statistics(engine, n_paths, params, policy):
    replay = engine._replay(n_paths, policy)
    count = len(engine.prepared.events)

    def payoff(p, scenario, ctx, initial):
        records, pv = lower_records(engine.prepared, ctx, policy)(
            initial, regression_scenario(engine.prepared, scenario), p["script"]
        )
        continuation = jax.vmap(policy.predict)(jnp.arange(count), records.features)
        exercise = (
            policy.exercise_mask
            & (records.conditions > 0.0)
            & (records.exercise_values > 0.0)
            & (records.exercise_values > continuation)
        )
        return jnp.concatenate(
            (jnp.stack((pv, pv * pv)), exercise.astype(pv.dtype), records.errors.astype(pv.dtype))
        )

    product = replace(
        replay.product, payoff=payoff, payoff_names=tuple(str(i) for i in range(count + 2))
    )
    replay = _ReplayEngine(product, engine.model, engine.settings)
    price = replay.checked_pricer(n_paths)
    replicas = engine.settings.lsmc_rqmc_replicates
    if replicas is None:
        values, errors = jax.jit(price)(params)
        values, errors = values[None, :], errors[None, :]
    else:
        shifts = jnp.asarray(
            np.stack(
                [
                    digital_shifts(
                        replay.sim_dim,
                        scramble_key(True, engine.settings.lsmc_pricing_seed or 0, r),
                    )
                    for r in range(replicas)
                ]
            )
        )
        offsets = jnp.arange(replicas, dtype=jnp.int64) * n_paths
        values, errors = jax.jit(
            jax.vmap(lambda offset, shift: price(dict(params, _lsmc_random=(offset, shift))))
        )(offsets, shifts)
    for flag, message in zip(np.any(np.asarray(errors), axis=0), product.error_messages):
        if flag:
            raise script_error(message)
    result = np.asarray(values)
    if not np.isfinite(result).all():
        raise InvalidPayoff("non-finite simulation diagnostic")
    return result


def _uncertainty_template():
    return {
        "mode": "not_applicable",
        "scramble": "none",
        "replicate_count": 1,
        "training_seed": None,
        "pricing_seed": None,
        "training_paths": 0,
        "validation_paths": 0,
        "pricing_paths_per_replicate": 0,
        "pricing_paths_total": 0,
        "replicate_means": [],
        "replicate_mean_se": None,
        "payoff_dispersion_se": None,
    }


def _policy_uncertainty(settings, n_paths, values):
    train, validation, replicas = _path_counts(settings, n_paths)
    mean, square = values[:, :2].mean(axis=0)
    result = _uncertainty_template()
    result.update(
        mode="deterministic",
        training_paths=train,
        validation_paths=validation,
        pricing_paths_per_replicate=n_paths,
        pricing_paths_total=replicas * n_paths,
        payoff_dispersion_se=math.sqrt(max(0.0, square - mean * mean) / (replicas * n_paths)),
    )
    if replicas > 1:
        result.update(
            mode="rqmc_conditional_policy",
            scramble="sobol-digital-shift-splitmix64-v1",
            replicate_count=replicas,
            training_seed=settings.lsmc_training_seed or 0,
            pricing_seed=settings.lsmc_pricing_seed or 0,
            replicate_means=list(values[:, 0]),
            replicate_mean_se=float(np.std(values[:, 0], ddof=1) / math.sqrt(replicas)),
        )
    return result


def _feature_name(feature, bindings):
    if feature.name != "SPOT":
        return feature.name
    return bindings[0] if bindings else "SPOT()"


def _regressor_index(names):
    if len(names) == 1 and names[0] != "SPOT()" and not names[0].startswith("VAR["):
        return names[0]
    return None


def _exercise_row(prepared, event, fit, names, all_dates, values):
    powers = fit.powers or basis_powers(len(names), fit.degree)
    return {
        "event_id": all_dates.index(str(prepared.event_dates[event])),
        "date": str(prepared.event_dates[event]),
        "basis_degree": fit.degree,
        "basis": "Constant" if fit.degree == 0 else "NormalizedMonomial",
        "effective_rank": fit.rank,
        "solver": fit.solver,
        "fallback_reason": fit.fallback_reason or None,
        "validation_mse": fit.validation_mse,
        "regressor_index": _regressor_index(names),
        "regression_features": names,
        "normalization_means": list(fit.means),
        "normalization_sigmas": list(fit.sigmas),
        "basis_powers": [list(term) for term in powers],
        "num_cond_true_paths": fit.count,
        "num_coefficients": len(fit.coefficients),
        "coefficients": list(fit.coefficients),
        "degenerate": bool(fit.reason),
        "degenerate_reason": fit.reason or None,
        "exercise_rate": float(values[:, 2 + event].mean()),
    }


def _exercise_diagnostics(data, engine, values):
    prepared = engine.prepared
    all_dates = [event["date"] for event in json.loads(describe(data))["events"]]
    names = [
        _feature_name(feature, _model_bindings(prepared))
        for feature in prepared.regression_features
    ]
    events = (i for i, row in enumerate(prepared.events) if exercise_node(row) is not None)
    return [
        _exercise_row(prepared, event, fit, names, all_dates, values)
        for event, fit in zip(events, engine.regressions)
    ]


def simulation_explain(data, model, n_paths, valuation=None, simulation=None):
    settings = simulation or MonteCarloSettings()
    if settings.enable_aad:
        raise script_error(
            "UnsupportedExecutionMode: the simulation diagnostic requires enable_aad=False"
        )
    prepared = prepare(data, model=model, valuation=valuation)
    result = {
        "schema": "dal.script-simulation/1",
        "evaluation_date": str(prepared.evaluation_date),
        "simulation": simulation_settings(settings),
        "n_paths": n_paths,
        "uncertainty": _uncertainty_template(),
        "exercise_events": [],
    }
    if prepared.expired or not prepared.has_exercise:
        if not prepared.expired:
            prepared.engine(model, settings).layout(n_paths)
        return result
    engine = prepared.engine(model, settings)
    params = engine.default_params()
    policy = engine.train(n_paths, params)
    values = _replay_statistics(engine, n_paths, params, policy)
    result["uncertainty"] = _policy_uncertainty(settings, n_paths, values)
    result["exercise_events"] = _exercise_diagnostics(data, engine, values)
    return result
