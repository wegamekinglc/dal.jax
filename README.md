# dal.jax

A re-implementation of [**DAL**](https://github.com/wegamekinglc/Derivatives-Algorithms-Lib)'s Monte Carlo pricing in pure Python (≥ 3.13) and [**JAX**](https://github.com/jax-ml/jax). Greeks come from `jax.grad` instead of an AAD tape. The same code runs on CPU and GPU, and paths are spread over devices with `shard_map`.

The full plan is in [issue #1](https://github.com/wegamekinglc/dal.jax/issues/1). This package implements **milestone P0**: the Monte Carlo engine, Sobol and pseudo-random numbers, the Brownian bridge, and the Black-Scholes model. Products are hand-written single-path payoffs for now. The script engine (DAL's event tables) is the next milestone. The notebooks in `jax/` and `xad/` are the earlier experiments.

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

In fuzzy mode (`enable_aad=True`, or `pricer(..., fuzzy=True)`) a payoff should smooth its discontinuities with DAL's kernels in `dal_jax.script.lower` (`cspr`, `bfly`). If it doesn't, `jax.grad` misses the barrier term. `tests/support.py` and `examples/02_barrier_option.ipynb` have an up-and-out call that matches DAL's fuzzy `d_BARRIER` and `d_vol`.

## Examples

[`examples/`](examples/) has four runnable notebooks: getting started with a European option, barrier Greeks and fuzzy smoothing, composing the pricer with JAX transforms, and multi-device execution and precision options. Run `uv sync --group examples`, then `uv run --group examples --with jupyterlab jupyter lab examples/`.

## Settings

`MonteCarloSettings` mirrors DAL's `MonteCarloSettings_` and adds the execution fields:

| field | default | meaning |
|---|---|---|
| `rsg` | `"sobol"` | `sobol` (gives the same points as DAL), `mrg32` / `irn` (`jax.random`, matches DAL statistically only) |
| `use_bb`, `enable_aad`, `smooth` | `False`, `False`, `0.01` | as in DAL |
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
uv run pytest          # 109 tests, ~2 min; 4 virtual CPU devices
```

`tests/oracle` compares against dal-python on identical Sobol points. Uniforms are bitwise equal and normals agree to 1e-14. European and monthly-barrier PV and Greeks agree with DAL to ~1e-12 or better, exact and fuzzy, with and without the bridge (the tolerances in issue #1 are 1e-10 for PV and 1e-8 for Greeks). `tests/mc/test_parallel.py` checks that 1 and 4 devices, all strategies, and different block sizes agree to 1e-13, and that `deterministic_reduction` gives bitwise-identical results on 1, 2, 3 and 4 devices.

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

Prices are on par with DAL. Gradients of long hand-unrolled schedules are slower: on XLA:CPU the reverse pass of a long chain of small elementwise ops costs far more than the forward pass. The script layer's scan grouping (P3) and the performance work in P4 address this.

## Layout

```
src/dal_jax/
  config.py           x64, virtual CPU devices, PRNG implementation, compilation cache
  errors.py           DAL-named exceptions (InvalidSetting, InvalidPathCount, ...)
  random/             sobol (+ directions.npy), inverse_normal, bridge, prng
  models/             base (protocol, SampleDef, Scenario), bs
  mc/                 settings, engine, parallel
  script/lower/       smoothing kernels (CSpr / BFly) shared with the future script layer
examples/             runnable notebooks (see examples/README.md)
scripts/              export_sobol_directions.py (regenerates the table from DAL's sobol.cpp)
benchmarks/           bench_mc.py
tests/                random/, models/, mc/, oracle/ (dal-python)
```
