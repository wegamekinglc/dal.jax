# Pricing with scripts

Start with an event table, a model and a valuation context. Preparation resolves the schedule and historical observations once; an engine then prices future paths. All executable examples follow this route.

## Price a contract and its risks

```python
import dal_jax as dj
from dal_jax.api import Product_New
from dal_jax.dates import Date

today = Date.ymd(2026, 10, 8)
context = dj.ValuationContext(evaluation_date=today)
product = Product_New(
    ["K", today.add_days(365)],
    ["100", "call PAYS MAX(SPOT()-K,0)"],
)
model = dj.BlackScholes(spot=100.0, vol=0.2, rate=0.03)
prepared = dj.prepare(product, valuation=context, model=model)
settings = dj.MonteCarloSettings(enable_aad=True)
engine = prepared.engine(model, settings)
result = engine.value(2**16)
```

The result contains `PV` and `d_<label>` risks for model and named script parameters. `enable_aad=True` selects fuzzy script conditions as well as differentiation. The smoothing width and kernel therefore affect both the valuation and its Greeks. Exact valuation is the default. [Example 02](../examples/02_barrier_option.py) shows a barrier script, and [example 06](../examples/06_scan_and_smoothing.py) compares event scans and smoothing kernels.

Definition rows contain constants or macros. Date rows contain assignments, conditions and payments; schedule rows expand into dated events. Vector operations support path-dependent state. `FIX(index)` observes a named equity, FX or rate index; `FIX(index, date)` observes a dated value. `PAYS ... ON date` requests a delayed payment. Pass `model` to preparation for observation and discount binding. [Examples 09–12](../examples/README.md) cover vectors, fixings, multiple equities and local volatility.

Pass `dal_jax.dates.Date` or `datetime.date` for event dates; a plain date string is interpreted as a definition name. Schedule strings support start/end, frequency, holiday centers and business-day adjustment; `DCF` supplies day-count fractions inside scripts. Built-in holiday centers are `CN.SSE`, `CN.IB` and `TARGET` (2000–2100). Space-separated centers combine calendars; an empty calendar has no holidays. [Example 05](../examples/05_script_and_history.py) demonstrates schedules and historical state.

## Supply historical observations

```python
past = today.add_days(-1)
context = dj.ValuationContext(
    evaluation_date=today,
    fixings=dj.FixingSnapshot({"EQ[A]": {past: 95.0}}),
)
product = Product_New(
    [past, today.add_days(365)],
    ["observed=FIX(EQ[A])", "pay PAYS observed"],
)
prepared = dj.prepare(product, valuation=context, model=model)
```

`FixingSnapshot` copies its input, canonicalizes index aliases and stores sorted immutable entries. Lookup preserves exact timestamps and reciprocal FX behavior. Date-only timestamps mean midnight; a midday quote does not satisfy a midnight request. Zero is a valid equity fixing. Missing historical observations fail during preparation.

Events strictly before the evaluation date are historical. Today's observations use model values by default; select `TodayFixingPolicy.REQUIREHISTORICAL` to require history for today as well. Historical events update state using hard conditions without accumulating past payments. A fully expired product returns zero price and risks without simulating the model. Legacy `SPOT()` history may be supplied with `historical_spots={date: value}`.

Contexts are independent of later input or session changes. For a service that replaces valuation settings, keep an explicitly owned session and capture one snapshot per request:

```python
session = dj.ValuationSession(context)
captured = session.snapshot()
session.update(dj.ValuationContext(evaluation_date=today.add_days(1)))
prepared = dj.prepare(product, valuation=captured, model=model)
```

`snapshot()` atomically captures the session's date, history and policy. If an unconfigured session needs a default date, it obtains today's host date at each snapshot. Supply a fixed date for reproducible requests. Separate sessions do not share business state.

## Reuse a pure pricing function

```python
import jax
import jax.numpy as jnp

params = engine.default_params()
price = engine.pricer(2**16)
pv, risks = jax.jit(jax.value_and_grad(lambda p: price(p)[0]))(params)
ladder = jax.vmap(
    lambda spot: price(params | {"model": params["model"] | {"spot": spot}})[0]
)(jnp.linspace(80.0, 120.0, 9))
```

The parameter tree has `model` and `script` groups. Build the pricing function before applying transforms; the returned function performs pure numerical computation. `engine.default_params()` places parameters on the selected devices. `engine.value()` also validates host inputs, checks script error flags and converts results for reporting; it is a host API.

Engine configuration is frozen because compiled functions capture settings, product plans, devices and random streams. Create a new engine to change configuration:

```python
from dataclasses import replace

other = prepared.engine(model, replace(settings, smooth=0.1))
engine.clear_cache()
```

`clear_cache()` releases that engine's references to pricing and record-collection functions. Previously returned functions and results remain usable. It does not reset JAX's process-wide compilation cache. Engine caches have no automatic capacity limit: reuse a bounded set of path counts, clear the cache or discard the engine in services with changing workloads. `block_bucketing=True` can reduce recompilation for ordinary pricers. Create factories outside transformed functions, and do not clear caches concurrently with factory construction.

## Train and price an exercise policy

```python
bermudan = Product_New(
    ["K", today.add_days(180), today.add_days(365)],
    ["100", "EXERCISE MAX(K-SPOT(),0)", "EXERCISE MAX(K-SPOT(),0)"],
)
exercise_settings = replace(settings, lsmc_training_paths=4096, lsmc_validation_paths=1024)
exercise = dj.prepare(bermudan, valuation=context, model=model).engine(model, exercise_settings)
evaluated = exercise.evaluate(2**16)
values = evaluated.as_dict()
regressions = evaluated.training.regressions
replicate_prices = evaluated.replicate_means
fixed_policy_price = exercise.pricer(2**16, policy=evaluated.training.policy)
```

`evaluate()` returns an immutable `LsmcResult`; `train_result()` returns a `TrainingResult` with the policy and its regression diagnostics. Subsequent training, valuation and cache clearing do not overwrite these results. `value()` still returns a dictionary, and `train()` still returns a policy. Neither engine exposes mutable last-result fields. `as_dict()` creates a new reporting dictionary.

`pricer()` trains immediately when no policy is supplied, then holds that policy fixed. `value()` and `evaluate()` train at the supplied parameters on every call. Training, validation and pricing use disjoint Sobol intervals. RQMC adds independent digital-shift pricing replicas, with uncertainty conditional on the common trained policy. Frozen risks differentiate pricing with passive coefficients; `RetrainedBump` adds the policy-risk secant convention implemented by DAL. [The LSMC guide](p6-p7.md) explains regression guards, degree selection and those risk conventions.

To compose device training with JAX transforms, construct `train_policy = exercise.training_pricer(n_paths)` first, then transform that callable. Host training provides the DAL-compatible regression diagnostics; fixed-shape device training serves the retrained-policy calculation.

## Select a backend and inspect the plan

Configure virtual CPU devices before the first JAX operation:

```python
dj.config.configure(num_cpu_devices=4, compilation_cache_dir="/tmp/dal-jax-compile")
settings = dj.MonteCarloSettings(platform="cpu", enable_aad=True)
```

For NVIDIA hardware, install the corresponding `cuda12` or `cuda13` extra and select `platform="gpu"`. Float64 remains the compatibility default on either backend. Choose float32 after checking prices and risks for the actual product; narrow fuzzy conditions can amplify path-rounding error. [Example 08](../examples/08_gpu_and_precision.py) and [the performance guide](performance.md) describe precision and block sizing.

Use `Product_DebugJson(product, valuation=context)` or `Product_DebugTree(product, valuation=context)` for DAL-compatible front-end strings. Those older dumps cover the scalar front-end vocabulary. `ScriptValuation_Explain(product, model, valuation=context)` returns a dictionary describing preparation, observation binding and historical replay for newer products. `ScriptSimulation_Explain(..., valuation=context, simulation=settings)` adds simulation and exercise diagnostics. [Example 15](../examples/15_gsr_rates.py) displays model observation slots.

## Migrate legacy stateful calls

`EvaluationDate_Set`, `EvaluationDate_Get` and `set_global_fixings` remain available with `DeprecationWarning`. They share one process-wide compatibility session. Atomic snapshots prevent torn date/history reads, but separate setter calls still cannot isolate concurrent requests. Migrate to a complete `ValuationContext`, including an explicit empty snapshot when no history is intended; a context supplies that empty snapshot by default.

`ValuationSettings` remains a partial compatibility configuration. Missing date or fixing fields are resolved from the legacy session once at the preparation boundary. Supplying only an explicit date does not opt out of legacy fixing fallback. `ValuationContext` requires a date and a non-null fixing snapshot, so it provides full isolation.

Use `MonteCarlo_ValueWithSettings(..., valuation=context, simulation=settings)` when retaining DAL API names. Replace reads of `engine.regressions` and `engine.replicate_means` with `engine.evaluate(...).training.regressions` and `.replicate_means` from the same result. Replace engine attribute assignment with construction using `dataclasses.replace(settings, ...)`.
