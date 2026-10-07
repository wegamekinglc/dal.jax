# dal.jax

A re-implementation of [**DAL**](https://github.com/wegamekinglc/Derivatives-Algorithms-Lib)'s Monte Carlo pricing in pure Python (≥ 3.13) and [**JAX**](https://github.com/jax-ml/jax). Greeks come from `jax.grad` instead of an AAD tape. The same code runs on CPU and GPU, and paths are spread over devices with `shard_map`.

The full plan is in [issue #1](https://github.com/wegamekinglc/dal.jax/issues/1). This package implements:

- **milestone P0**: the Monte Carlo engine, Sobol and pseudo-random numbers, the Brownian bridge, and the Black-Scholes model. Products are hand-written single-path payoffs.
- **milestone P1**: the front end of DAL's script engine. It reads event tables (macros, constants, schedules, holidays, day counts, index names), parses the script language and runs DAL's analysis passes. Its `Product_Describe`, `Product_DebugJson`, `Product_DebugTree` and `Product_Debug` output is byte-identical to dal-python.
- **milestone P2**: exact scalar script valuation: arithmetic, functions, IF and PAYS lower to JAX event functions, with historical replay and the `BSModelData_New` / `MonteCarlo_Value` compatibility API. European, scalar Asian and autocall prices match DAL on identical Sobol paths.
- **milestone P3**: fuzzy scalar scripts and all model/script Greeks, safe nested IF blending, adjacent event scan grouping, and optional C1 smoothing. Barrier and autocall sensitivities match DAL on identical Sobol paths.

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

In fuzzy mode (`enable_aad=True`, or `pricer(..., fuzzy=True)`) hand-written payoffs should smooth their discontinuities with DAL's kernels in `dal_jax.script.lower` (`cspr`, `bfly`). If they don't, `jax.grad` misses the barrier term. Script products apply smoothing automatically. `tests/support.py` and `examples/02_barrier_option.ipynb` have an up-and-out call that matches DAL's fuzzy `d_BARRIER` and `d_vol`.

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

Valuation currently supports the scalar subset, including expanded `FOR` loops and predefined vector entries that the parser turns into constants. Mutable vectors, `FIX`, delayed `PAYS ... ON` and `EXERCISE` raise explicit preparation or execution errors; their valuation belongs to P5/P6. A payment explicitly made `ON` its own event date is treated as an ordinary `PAYS`.

## Examples

[`examples/`](examples/) has four runnable notebooks: getting started with a European option, barrier Greeks and fuzzy smoothing, composing the pricer with JAX transforms, and multi-device execution and precision options. Run `uv sync --group examples`, then `uv run --group examples --with jupyterlab jupyter lab examples/`.

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
| `block_size` | `8192` | paths per block; the last block is masked, so it never forces a recompile |
| `parallel` | `"shard_map"` | also `auto` (GSPMD), `pmap` (for comparison only), `none` |
| `platform`, `devices` | `"auto"`, all | `cpu` / `gpu`, or an explicit device tuple |
| `dtype` | `"float64"` | `float32` casts the path arrays; block sums are still added in float64 |
| `deterministic_reduction` | `False` | sums per-block values and gradients in block order, so results are bitwise identical for any device count (reverse mode only) |
| `block_bucketing` | `False` | rounds the block count up to a power of two, so nearby `n_paths` reuse one compilation |
| `checkpoint` | `True` | recomputes each block in the backward pass, so gradient memory is one block's worth |

On CPU, split the host into virtual devices before running any JAX operation:

```python
import dal_jax as dj
dj.config.configure(num_cpu_devices=8, compilation_cache_dir="~/.cache/dal_jax")
```

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
uv run pytest          # 572 tests; 4 virtual CPU devices
```

`tests/oracle` compares against dal-python on identical Sobol points. Uniforms are bitwise equal and normals agree to 1e-14. European and monthly-barrier PV and Greeks agree with DAL to ~1e-12 or better, exact and fuzzy, with and without the bridge (the tolerances in issue #1 are 1e-10 for PV and 1e-8 for Greeks). `tests/mc/test_parallel.py` checks that 1 and 4 devices, all strategies, and different block sizes agree to 1e-13, and that `deterministic_reduction` gives bitwise-identical results on 1, 2, 3 and 4 devices.

For the script front end, `tests/oracle/test_dal_script_frontend.py` checks that the four dumps are byte-identical to dal-python on a corpus of event tables, and that malformed scripts and definitions fail with the same error. `tests/script`, `tests/dates` and `tests/test_index.py` port DAL's C++ unit tests for the lexer, parser, passes, events, dates and index names. These cover the features newer than dal-python as well.

`tests/oracle/test_dal_script_prices.py` checks exact script prices with and without the bridge, including European calls/puts, a scalar Asian, an autocall, nested conditions, historical replay, DCF schedules, same-date events and expired products. `tests/script/test_exact.py` checks finite prices and gradients through unused invalid branches; `tests/test_api_value.py` covers the compatibility API and today's zero-dimensional model path across every parallel strategy.

`tests/oracle/test_dal_script_greeks.py` checks fuzzy script PV and all model/script Greeks, the million-path barrier reference, and common-path finite differences. `tests/script/test_eventgroup.py` checks scan/unrolled agreement to 1e-14 and constant graph size as a daily schedule grows from 36 to 365 observations. Script tests also cover inactive invalid arithmetic, nested fractional states, tiny divisors, historical parameter risks, C1 joins, JAX transforms and multi-device scans.

## Benchmarks

`python benchmarks/bench_mc.py --devices 8 --dal` runs 2²⁰ Sobol paths. The table below is warm wall time on a 32-thread laptop under WSL2. DAL runs 32 threads on the same machine.

| case | JAX, 1 CPU device | JAX, 8 CPU devices | DAL |
|---|---:|---:|---:|
| European, price | 23 ms | 4.8 ms | 4.4 ms |
| European, price + 5 Greeks | 35 ms | 11 ms | 16 ms |
| barrier 1M (36 dates), price | 189 ms | 143 ms | 151 ms |
| barrier 1M, price + 6 Greeks | 955 ms | 560 ms | 393 ms |
| barrier 1W (156 dates), price | 890 ms | 576 ms | 655 ms |
| barrier 1W, price + 6 Greeks | 6.1 s | 7.4 s | 1.6 s |

Prices are on par with DAL. Gradients of long hand-unrolled schedules are slower: on XLA:CPU the reverse pass of a long chain of small elementwise ops costs far more than the forward pass. Script products now group repeated events into scans; P4 covers broader CPU/GPU tuning.

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
  models/             base (protocol, SampleDef, Scenario), bs
  mc/                 settings, engine, parallel
  script/             lexer, preprocessor, parser, ast, product, preparation, debug, diagnostics
  script/passes/      varindex, ifmeta, constfold, domain (+ intervals), constcond, eventgroup
  script/lower/       exact/fuzzy scalar events, grouped execution, DAL and C1 smoothing kernels
examples/             runnable notebooks (see examples/README.md)
scripts/              export_sobol_directions.py, export_calendars.py (regenerate data from DAL)
benchmarks/           bench_mc.py, bench_script_compile.py (+ CPU compilation report)
tests/                random/, models/, mc/, script/, dates/, oracle/ (dal-python)
```
