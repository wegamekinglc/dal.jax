# dal.jax

A re-implementation of [**DAL**](https://github.com/wegamekinglc/Derivatives-Algorithms-Lib)'s Monte Carlo pricing in pure Python (≥ 3.13) and [**JAX**](https://github.com/jax-ml/jax). Greeks come from `jax.grad` instead of an AAD tape. The same code runs on CPU and GPU, and paths are spread over devices with `shard_map`.

The full plan is in [issue #1](https://github.com/wegamekinglc/dal.jax/issues/1). This package implements:

- **milestone P0**: the Monte Carlo engine, Sobol and pseudo-random numbers, the Brownian bridge, and the Black-Scholes model. Products are hand-written single-path payoffs.
- **milestone P1**: the front end of DAL's script engine. It reads event tables (macros, constants, schedules, holidays, day counts, index names), parses the script language and runs DAL's analysis passes. Its `Product_Describe`, `Product_DebugJson`, `Product_DebugTree` and `Product_Debug` output is byte-identical to dal-python.
- **milestone P2**: exact scalar script valuation: arithmetic, functions, IF and PAYS lower to JAX event functions, with historical replay and the `BSModelData_New` / `MonteCarlo_Value` compatibility API. European, scalar Asian and autocall prices match DAL on identical Sobol paths.
- **milestone P3**: fuzzy scalar scripts and all model/script Greeks, safe nested IF blending, adjacent event scan grouping, and optional C1 smoothing. Barrier and autocall sensitivities match DAL on identical Sobol paths.
- **milestone P4**: CPU device and block tuning, actual CUDA validation, GPU precision and memory-based block sizing, repeatable RBG streams under parallel transforms, and an end-to-end benchmark suite.
- **milestone P5**: mutable vector valuation, dated `FIX` and immutable fixing snapshots, delayed payments, correlated Black-Scholes with factor-aware bridges, and local volatility with every surface-node risk.
- **milestone P6**: three-phase LSMC exercise valuation, guarded normalized regression, held-out degree selection, RQMC replicas and frozen/retrained-policy risks.
- **milestone P7**: single/multi-factor GSR, GSRSLV, domestic equity/rate hybrids and preparation/simulation diagnostics. The `0.1.0a1` release package is prepared; PyPI upload is deferred.

## Install

```bash
uv venv --python 3.13 && uv pip install -e .   # or: pip install -e ".[cuda12]" for NVIDIA GPUs
uv sync                                         # + dev tools: pytest, scipy, dal-python (the oracle)
```

## Quick start

```python
import jax.numpy as jnp
import dal_jax as dj   # enables jax_enable_x64

def call(params, scenario, ctx):                      # one path; the engine vmaps it
    payoff = jnp.maximum(scenario.spot[-1] - params["script"]["STRIKE"], 0.0)
    return payoff / scenario.numeraire[-1]            # numeraire-deflated, like DAL's PAYS

product = dj.PathProduct(timeline=(3.0,), payoff=call, script_params={"STRIKE": 120.0})
model = dj.BlackScholes(spot=100.0, vol=0.15, rate=0.05, div=0.03)

engine = dj.MonteCarloEngine(product, model, dj.MonteCarloSettings(enable_aad=True))
engine.value(2**20)
# {'PV': 5.201768833210826, 'd_spot': 0.33503144936845, 'd_vol': 59.5850327569119,
#  'd_rate': 84.9041283109051, 'd_div': -100.509434810537, 'd_STRIKE': -0.235844800863624}
```

DAL's `MonteCarlo_Value(product, BSModelData_New(100, .15, .05, .03), 2**20, "sobol", False, True)` returns the same PV and Greeks for the equivalent event table, within a few ulps.

### The JAX-native layer

`engine.pricer(n_paths)` returns a pure `f(params) -> pv[n_payoffs]`. You can apply `jit`, `grad`, `jacrev`, `jacfwd`, `hessian` and `vmap` to it:

```python
import jax

f = engine.pricer(2**18)
params = engine.default_params()               # {"model": {spot, vol, rate, div}, "script": {STRIKE}}
pv, grads = jax.value_and_grad(lambda p: f(p)[0])(params)
ladder = jax.vmap(lambda s: f({**params, "model": params["model"] | {"spot": s}})[0])(jnp.linspace(80, 120, 9))
```

In fuzzy mode (`enable_aad=True`, or `pricer(..., fuzzy=True)`) hand-written payoffs should smooth their discontinuities with DAL's kernels in `dal_jax.script.lower` (`cspr`, `bfly`). If they don't, `jax.grad` misses the barrier term. Script products apply smoothing automatically. `tests/support.py` and `examples/02_barrier_option.py` have an up-and-out call that matches DAL's fuzzy `d_BARRIER` and `d_vol`.

### Script products

`dal_jax.api` has dal-python's names and signatures, so event tables move over unchanged. A string cell is a definition or a schedule, and a date cell is an event date:

```python
from dal_jax.api import BSModelData_New, EvaluationDate_Set, MonteCarlo_Value, Product_New, Product_DebugTree, Product_Describe
from dal_jax.dates import Date

EvaluationDate_Set(Date.ymd(2026, 10, 7))
product = Product_New(
    ["STRIKE", "BARRIER", Date.ymd(2026, 10, 7), "START: 2026-10-07 END: 2027-01-07 FREQ: 1M", Date.ymd(2027, 1, 7)],
    ["120", "150", "alive = 1", "IF spot() >= BARRIER:0.1 THEN alive = 0 END", "call PAYS alive * MAX(spot() - STRIKE, 0)"],
)
print(Product_DebugTree(product))
```

```
Variables: alive, call*
Constants: BARRIER=150, STRIKE=120

📅 1 · 2026-10-07 · future
└── (1) alive ← 1

📅 2 · 2026-11-07 · future
└── (1) if spot() ≥ BARRIER ⟨ε=0.1⟩ then alive ← 0

📅 3 · 2026-12-07 · future
└── (1) if spot() ≥ BARRIER ⟨ε=0.1⟩ then alive ← 0

📅 4 · 2027-01-07 · future
├── (1) if spot() ≥ BARRIER ⟨ε=0.1⟩ then alive ← 0
└── (2) call ⇐ alive × max(spot() − STRIKE, 0)
```

Price the same event table through the compatibility API:

```python
model = BSModelData_New(100.0, 0.15, 0.05, 0.03)
result = MonteCarlo_Value(product, model, 2**16, "sobol", use_bb=True, enable_aad=True)
print(result)  # PV, d_spot, d_vol, d_rate, d_div, d_BARRIER, d_STRIKE
```

The API accepts `method` as an alias for `rsg`, an explicit `evaluation_date`, and execution options from `MonteCarloSettings` (`block_size`, `parallel`, `devices`, `dtype`, ...). `compiled=True/False` is accepted with a warning: XLA always compiles the payoff. With `enable_aad=True`, future IF conditions use DAL's fuzzy kernels and the result includes every `d_<label>`.

For repeated pricing or JAX transforms, prepare once and keep the engine:

```python
from dal_jax import MonteCarloEngine, prepare
from dal_jax.api import EvaluationDate_Get

prepared = prepare(product, EvaluationDate_Get())  # immutable, hashable event and observation plan
engine = MonteCarloEngine(prepared.path_product(), model)
price = engine.value(2**16)
f = engine.pricer(2**16)                          # exact scalar script payoff
```

Past events run once on the host with hard conditions; their `PAYS` expressions are evaluated without accumulating payments. Every future path starts from that state. To replay past `SPOT()` calls, pass `historical_spots={event_date: value}` to `prepare` or `MonteCarlo_Value`; missing values raise `UnboundHistoricalSpot`. Historical assignments depending on named script parameters also have a differentiable JAX replay: standard reduction computes it once per device, outside the path and block loops. Deterministic reduction replays it per block for independent block Jacobians. A fully expired product returns zero PV and, when requested, zero Greeks without model allocation or simulation.

Adjacent events with the same normalized structure share one `lax.scan` body. Date literals, folded counters and DCF values become per-event constant slots; named parameters remain differentiable. `scan_group_threshold=4` is the default, and `0` disables scanning. `prepared.event_groups(fuzzy=True)` exposes each group's span, template and constants. Fuzzy preparation retains continuous comparisons as DAL's model-aware preparation does; the lowerer also supports discrete `lb/rb` metadata from the legacy domain pass.

DAL's piecewise-linear kernels remain the default. `smoothing_kernel="smoothstep"` selects bounded cubic C1 transitions, including a zero-slope butterfly peak, for second derivatives. This option changes the smoothing profile and its Greeks.

The front end follows DAL's current `master`, so it also supports what dal-python 2026.9.25 lacks: vectors (`APPEND`, `v[i]`, `SUM`/`AVERAGE`/`MIN`/`MAX`, predefined `[1, 2, 3]` definitions), `FOR` loops over constant bounds, `PAYS ... ON date`, `FIX(index, date)` observations, `EXERCISE` statements, IR index names and the `30U/360` basis. Errors raise the `DalError` subclass that DAL names in its message (`InvalidIndex`, `DuplicateExercise`, ...), with the same text.

Valuation supports mutable vectors, expanded `FOR`, reductions, dated `FIX` and delayed payments. Vectors use fixed capacities and dynamic lengths; fuzzy branches blend padded values and use the longer length only inside the transition. Invalid reads and empty non-SUM reductions raise DAL-named exceptions through `engine.value`. For pure JAX code, `engine.checked_pricer(n_paths)` returns `(prices, error_rates)`; `pricer` returns NaN when a live path has a vector error.

`prepare(..., model=model, valuation=ValuationSettings(...))` binds index observations and discount maturities before folding branches. `FixingSnapshot` copies historical quotes and supports exact timestamp lookup and inverse FX quotes. Today's default policy uses the model; `RequireHistorical` requires today's snapshot quote. Missing historical quotes do not fall back to the model. A `default_index` in `ScriptProductSettings` binds legacy `SPOT()` when using FIX or multiple assets. Future plain equity observations are supported by BS, correlated BS and local vol; historical equity/FX/Libor fixings can come from snapshots. GSR and hybrid models support future domestic rate observations; future FX models are outside this alpha. Past events with an unsettled delayed payment raise `UnsettledDelayedPayment`.

`CorrelatedBlackScholes` takes equity index names, spots, volatilities, dividends and a positive-definite correlation matrix. Its per-asset parameters are differentiable; correlation is passive. `LocalVol` uses log-spot/time bilinear interpolation with flat extrapolation, subdivides each product time gap by `max_step`, and runs checkpointed log-Euler steps. Its surface replaces the scalar BS volatility; `vol_grid(gradient_model_params)` reconstructs the bucket-vega matrix. See [P5 usage and verification](docs/p5.md) for those models. Exercise valuation, single/multi-factor GSR, GSRSLV, domestic hybrids and both valuation/simulation diagnostics are covered in [P6/P7 methods and migration](docs/p6-p7.md). For exercise scripts, bind the model with `prepare(..., model=model)` and use `prepared.engine(model, settings)`; this selects the LSMC engine automatically.

## Examples

[`examples/`](examples/README.md) has sixteen ordinary Python scripts covering European and barrier pricing, JAX transforms, parallel execution, historical scripts and discounts, event scans and C1 smoothing, PRNG streams, CPU/GPU precision, vectors, fixing policies, correlated assets, local volatility, Bermudan exercise, RQMC, rates and hybrids. Every example prints numerical and synchronized performance comparisons with dal-python. Run `uv sync --group examples`, then `uv run --group examples python examples/01_european_option.py`. Saved numerical and timing reports are in `examples/results/`. Examples 09–16 need the pinned source-built dal-python oracle; installation commands are in the examples README.

## Settings

`MonteCarloSettings` mirrors DAL's `MonteCarloSettings_` and adds the execution fields:

| field | default | meaning |
|---|---|---|
| `rsg` | `"sobol"` | `sobol` (gives the same points as DAL), `mrg32` / `irn` (`jax.random`, matches DAL statistically only) |
| `use_bb`, `enable_aad`, `smooth` | `False`, `False`, `0.01` | as in DAL |
| `smoothing_kernel` | `"dal"` | DAL's linear CSpr/BFly; `"smoothstep"` selects cubic C1 kernels |
| `scan_group_threshold` | `4` | minimum adjacent equal event count for a script scan; `0` disables scanning |
| `inverse_normal` | `"acklam"` | DAL's `InverseNCDF`; `acklam_polish[_precise]` or `ndtri` for more accuracy |
| `sobol_shift_key` | `None` | DAL's digital shift (SplitMix64) |
| `block_size` | `"auto"` | CPU uses 8192; GPU estimates a power-of-two block from allocator memory and the product's horizon; a positive integer overrides it |
| `parallel` | `"shard_map"` | also `auto` (GSPMD), `pmap` (for comparison only), `none` |
| `platform`, `devices` | `"auto"`, all | `cpu` / `gpu`, or an explicit device tuple |
| `dtype` | `"float64"` | `float32` casts the path arrays; `auto` chooses float64 on CPU and float32 on GPU; block sums are added in float64 |
| `deterministic_reduction` | `False` | sums per-block values and gradients in block order, so results are bitwise identical for any device count (reverse mode only) |
| `block_bucketing` | `False` | rounds the block count up to a power of two, so nearby `n_paths` reuse one compilation |
| `checkpoint` | `True` | rematerializes blocks and grouped script event steps in the backward pass |

On CPU, split the host into virtual devices before running any JAX operation:

```python
import dal_jax as dj
dj.config.configure(num_cpu_devices=8, compilation_cache_dir="~/.cache/dal_jax")
```

For NVIDIA GPUs, install the matching JAX extra (`.[cuda12]` or `.[cuda13]`), then select the backend explicitly:

```python
settings = dj.MonteCarloSettings(platform="gpu", dtype="float64", enable_aad=True)
engine = dj.MonteCarloEngine(prepared.path_product(), model, settings)
result = engine.value(2**20)
```

The compatibility default stays float64 on both platforms. Choose float32 explicitly, or with `dtype="auto"`, after checking your product's error. The path reduction uses JAX's float32 reduction; blocks accumulate in float64. The million-path autocall benchmark has large float32 Greek errors at the default narrow smoothing width, despite an accurate PV; use float64 for those risks. See the measured errors, memory policy and recommendations in [the performance report](docs/performance.md).

`prng_impl="rbg"` is an optional GPU choice for `mrg32`/`irn`. RBG block generation maps keys sequentially when batched, preserving each block's stream under GSPMD and nested `vmap`. With a fixed block size, device count and parallel strategy do not change the stream. Changing block size changes PRNG paths; Sobol paths depend only on global path id. `value` places caller parameters on the selected device mesh; native transforms should start from `engine.default_params()`.

## How the engine works

```
global path id ──► Sobol point id+1 (or fold_in(key, block)) ──► InverseNCDF ──► [bridge]
   ──► model.generate (vmap over a block) ──► payoff ──► masked block sum
   ──► lax.scan over a device's blocks (jax.checkpoint per block) ──► psum over devices
```

- **Sobol**: the DAL direction table (21201 × 32, `random/directions.npy`), random access by Gray code. Any device can compute any path, so there is no `SkipTo`.
- **Model protocol** (`models/base.py`): `allocate` runs on the host and returns a static plan. `init` depends only on the parameters and runs once per pricing, so gradients flow through it. `generate` handles a single path.
- **Parallelism**: each device scans a contiguous range of blocks, and its block ids come from `axis_index`. Replicated parameters are cast to device-varying once on entry. Without that cast, every block adds a cross-device `psum` to the backward pass, and the gradient stops scaling with devices.

## Parity and tests

```bash
uv run pytest          # CPU suite; 4 virtual CPU devices, optional GPU tests skipped
# In an environment with a CUDA/ROCm-enabled JAX:
uv run pytest tests/gpu --run-gpu
```

`tests/oracle` compares against dal-python on identical Sobol points. Uniforms are bitwise equal and normals agree to 1e-14. European and monthly-barrier PV and Greeks agree with DAL to ~1e-12 or better, exact and fuzzy, with and without the bridge (the tolerances in issue #1 are 1e-10 for PV and 1e-8 for Greeks). `tests/mc/test_parallel.py` checks that 1 and 4 devices, all strategies, and different block sizes agree to 1e-13, and that `deterministic_reduction` gives bitwise-identical results on 1, 2, 3 and 4 devices.

For the script front end, `tests/oracle/test_dal_script_frontend.py` checks that the four dumps are byte-identical to dal-python on a corpus of event tables, and that malformed scripts and definitions fail with the same error. `tests/script`, `tests/dates` and `tests/test_index.py` port DAL's C++ unit tests for the lexer, parser, passes, events, dates and index names. These cover the features newer than dal-python as well.

`tests/oracle/test_dal_script_prices.py` checks exact script prices with and without the bridge, including European calls/puts, a scalar Asian, an autocall, nested conditions, historical replay, DCF schedules, same-date events and expired products. `tests/script/test_exact.py` checks finite prices and gradients through unused invalid branches; `tests/test_api_value.py` covers the compatibility API and today's zero-dimensional model path across every parallel strategy.

`tests/oracle/test_dal_script_greeks.py` checks fuzzy script PV and all model/script Greeks, the million-path barrier reference, and common-path finite differences. `tests/script/test_eventgroup.py` checks scan/unrolled agreement to 1e-14 and constant graph size as a daily schedule grows from 36 to 365 observations. Script tests also cover inactive invalid arithmetic, nested fractional states, tiny divisors, historical parameter risks, C1 joins, JAX transforms and multi-device scans.

## Benchmarks

`uv run python benchmarks/bench_suite.py --devices 8 --dal --output cpu.json` runs the five scalar script products with 2²⁰ paths, exact prices and fuzzy prices plus all Greeks. It records lowering, compilation, synchronized warm times, results and memory estimates. GPU runs select `--platform gpu --dtype float64` or `float32`.

The [P6/P7 performance report](docs/p6-p7-performance.md) adds million-path Bermudan phase timings on CPU 1/4 and GPU, plus rate/SLV/hybrid comparisons. The [P4 performance report](docs/performance.md) contains CPU 1/4/8/16/32 device runs, block and strategy comparisons, checkpoint storage, actual RTX 4060 Laptop GPU results, DAL comparisons and the limits of float32 Greeks. On this machine CPU 8 balances the long workloads; GPU auto sizing uses 32768 price paths per block and 8192–32768 for Greeks. Narrow autocall conditions require float64 for reliable Greeks. Raw reports are in `benchmarks/p4_*.json`; `bench_mc.py` remains the earlier hand-written payoff benchmark.

`uv run python benchmarks/bench_script_compile.py --output benchmarks/script_compile_cpu.json` isolates compilation of script price and parameter gradients from path generation. It reports graph equation counts, StableHLO size, lowering time and compilation time for grouped and unrolled daily schedules. The checked-in JSON records a CPU run; timings vary by machine.

## Layout

```
src/dal_jax/
  config.py           x64, virtual CPU devices, PRNG implementation, compilation cache
  errors.py           DAL-named exceptions (InvalidSetting, InvalidPathCount, ScriptError subclasses, ...)
  api.py              dal-python compatible Product_*, EvaluationDate_*, BSModelData_New, MonteCarlo_Value
  strings.py          DAL's case-insensitive strings and number parsing
  index.py            index names (EQ, FX, IR) and their canonical forms
  dates/              Date, increments, holidays (+ calendar_data.py), schedules, day bases
  random/             sobol (+ directions.npy), inverse_normal, bridge, prng
  models/             base (protocol, SampleDef, Scenario), bs, correlated_bs, localvol, gsr, gsrslv, hybrid
  mc/                 settings, engine, parallel, tuning, lsmc, regression, regression_device
  script/             lexer, preprocessor, parser, ast, product, preparation, observation, fixings, debug, diagnostics, explain, lsmcprep
  script/passes/      varindex, ifmeta, constfold, domain (+ intervals), constcond, eventgroup
  script/lower/       exact/fuzzy/LSMC events, state, grouped execution, smoothing kernels
examples/             Python scripts with DAL numerical/performance comparisons
scripts/              data exporters, build_dal_oracle.sh (pinned native reference)
benchmarks/           bench_mc.py, bench_script_compile.py, bench_suite.py (+ measured JSON reports)
tests/                random/, models/, mc/, script/, dates/, oracle/ (dal-python), gpu/ (opt-in)
```
