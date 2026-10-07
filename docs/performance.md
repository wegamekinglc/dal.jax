# CPU and GPU performance (P4)

Measured on 2026-10-07. Keep float64 for DAL-compatible Greeks. Eight virtual CPU devices are a useful starting point on this machine. GPU float32 is faster for most measured workloads, but the tightly smoothed autocall has inaccurate float32 Greeks despite an accurate price; its timing is marked below.

## Setup and measurement

- Host: Intel Core i9-13900HX, 32 logical processors visible to WSL2; Linux 5.15.167.4, glibc 2.43.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, driver 595.79, the `cuda13` extra. Only one physical GPU was available.
- Python 3.13.9, JAX/jaxlib 0.11.2; dal-python 2026.9.25 with 32 threads.
- GPU processes used `XLA_PYTHON_CLIENT_MEM_FRACTION=.35`; the allocator reported a 3,005,218,816-byte limit (about 2.80 GiB). JAX's allocator reservation is separate from the block-sizing budget. See [JAX's memory allocation documentation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).
- Each contract uses 2²⁰ paths, unshifted Sobol with DAL's float64 Acklam transform, no Brownian bridge, and a three-year BS model: spot 100, vol .15, rate .05, dividend .03. Event tables are in [products.py](../benchmarks/products.py). Monthly contracts have 36 simulated intervals; the weekly schedule plus explicit maturity has 157.
- Price is exact; price plus Greeks uses DAL fuzzy kernels and width .01. The P4 Asian uses a scalar running sum; vector valuation and its measurements are in the [P5 report](p5.md). Bermudan belongs to P6 and is excluded.
- [bench_suite.py](../benchmarks/bench_suite.py) prepares once, lowers and compiles the pure pricer separately, records the first synchronized execution, then synchronizes every warm execution with `jax.block_until_ready`. The tables use the minimum warm wall time in milliseconds. JSON files also contain medians, raw runs, results, resolved blocks and compiler memory estimates.
- CPU 1/4/8 and DAL use five warm repeats; the device/block/strategy studies and GPU runs use three. Timed benchmark processes ran sequentially. DAL uses its valuation API; JAX timings exclude preparation and `engine.value`'s host validation/result conversion. These are repeat-pricing measurements, not cold API latency. Laptop clock changes and WSL/Windows activity affect timings.

## End-to-end warm runs

| Product / mode | CPU 1, float64 | CPU 8, float64 | GPU float64 | GPU float32 | DAL, 32 threads |
|---|---:|---:|---:|---:|---:|
| European, price | 28.2 | 3.7 | 3.5 | 3.7 | 3.7 |
| European, price + Greeks | 47.6 | 8.4 | 6.4 | 5.1 | 11.8 |
| Monthly barrier, price | 282.2 | 101.6 | 106.8 | 87.1 | 112.5 |
| Monthly barrier, price + Greeks | 1295.5 | 415.9 | 283.3 | 213.1 | 282.8 |
| Weekly barrier, price | 1045.7 | 487.2 | 422.2 | 361.2 | 546.7 |
| Weekly barrier, price + Greeks | 3645.3 | 1841.6 | 1895.1 | 1442.1 | 1273.3 |
| Scalar Asian, price | 259.4 | 104.2 | 104.0 | 82.1 | 151.9 |
| Scalar Asian, price + Greeks | 843.3 | 326.8 | 258.8 | 199.5 | 326.6 |
| Autocall, price | 352.1 | 123.9 | 140.9 | 92.8 | 127.7 |
| Autocall, price + Greeks | 3971.5 | 1265.0 | 585.6 | 385.3 † | 320.5 |

† The float32 autocall Greeks at width .01 are inaccurate; this cell reports execution time only. Use the float64 column for comparable risks.

CPU uses block 8192 and `shard_map`. Final GPU auto sizing resolves every price block to 32768; Greek blocks are 32768 except the weekly barrier, which uses 8192. JSON baselines: [CPU 1](../benchmarks/p4_cpu_1.json), [CPU 4](../benchmarks/p4_cpu_4.json), [CPU 8](../benchmarks/p4_cpu_8.json), [GPU float64](../benchmarks/p4_gpu_64.json), [GPU float32](../benchmarks/p4_gpu_32.json).

GPU does not win every workload. Long reverse scans remain expensive, while the generic scalar autocall's CPU Greeks are substantially slower than DAL. This report does not claim that P4 removes those bottlenecks.

## CPU tuning

| Product / mode | 1 device | 4 devices | 8 devices | 16 devices | 32 devices |
|---|---:|---:|---:|---:|---:|
| European, price | 28.2 | 6.7 | 3.7 | 3.3 | 4.0 |
| European, greeks | 47.6 | 11.8 | 8.4 | 9.2 | 13.1 |
| Weekly barrier, price | 1045.7 | 580.9 | 487.2 | 491.7 | 517.4 |
| Weekly barrier, greeks | 3645.3 | 2124.7 | 1841.6 | 2166.8 | 2276.2 |
| Autocall, price | 352.1 | 151.6 | 123.9 | 128.2 | 136.0 |
| Autocall, greeks | 3971.5 | 1558.4 | 1265.0 | 1326.7 | 1560.2 |

Eight devices provide the best measured long-contract throughput. Sixteen devices slightly improve the European price but slow its Greeks; 32 devices add overhead. Device count must be set before the first JAX operation and remains a user choice rather than a machine-independent automatic default. Raw runs: [16 devices](../benchmarks/p4_cpu_16.json), [32 devices](../benchmarks/p4_cpu_32.json).

| CPU 8 strategy, price + Greeks | European | Monthly barrier |
|---|---:|---:|
| shard_map | 8.4 | 415.9 |
| auto / GSPMD | 27.0 | 716.6 |
| pmap | 12.7 | 541.0 |

`shard_map` remains the default. Automatic partitioning and `pmap` retain their correctness tests but were slower here. Raw runs: [GSPMD](../benchmarks/p4_cpu_auto.json), [pmap](../benchmarks/p4_cpu_pmap.json).

| CPU 8 block, price + Greeks | Weekly barrier | Autocall |
|---|---:|---:|
| 512 | 1890.6 | 1589.0 |
| 2048 | 2024.8 | 1292.5 |
| 8192 | 1841.6 | 1265.0 |

Smaller CPU blocks reduce storage but gave no throughput gain. Raw runs: [512](../benchmarks/p4_cpu_block_512.json), [2048](../benchmarks/p4_cpu_block_2048.json).

## GPU block sizing, compilation and memory

`block_size="auto"` keeps 8192 on CPU. GPU sizing uses 20% of the smallest selected device's allocator limit, divided by an array estimate: float64 normals plus path-dtype scenario slots/payoffs, multiplied by 16 for AAD or 4 for price. The result is rounded down to a power of two and clamped to 256–32768. Missing allocator information falls back to 8192. A positive integer overrides the estimate. This is a heuristic, not a bound on XLA memory, especially for additional business batch axes or disabled checkpointing.

The weekly float32 Greek block study measured 1639.1 ms at 8192 and 1008.3 ms at 32768. The latter uses 748.2 MiB of reported temporary buffers. Larger blocks can improve throughput when memory allows; auto keeps headroom. Raw runs: [8192](../benchmarks/p4_gpu_32_block_8192.json), [32768 and other contracts](../benchmarks/p4_gpu_32_block_32768.json).

| Backend | Lowering range (s) | XLA compilation range (s) | Largest reported temporary buffer (MiB) |
|---|---:|---:|---:|
| CPU 1 | 0.084–0.879 | 0.146–0.835 | 275.8 |
| CPU 8 | 0.077–0.844 | 0.171–0.997 | 275.8 |
| GPU float64 | 0.077–0.881 | 0.336–1.824 | 210.0 |
| GPU float32 | 0.077–0.853 | 0.363–1.895 | 187.0 |

These ranges cover all five contracts in both modes. `temp_size_in_bytes` is XLA's compiled-buffer estimate, not RSS, allocator pool reservation, or a measured per-contract peak. GPU `peak_bytes_in_use` in JSON is cumulative within the process, so later rows cannot be treated as independent peaks.

At 2¹⁸ monthly-barrier paths on CPU 8, checkpointing used 65.1 MiB of reported temporary buffers and 159.3 ms; disabling it used 122.4 MiB and 273.0 ms. Keep the default checkpoint. Raw runs: [enabled](../benchmarks/p4_cpu_checkpoint.json), [disabled](../benchmarks/p4_cpu_no_checkpoint.json).

Event scans keep compilation small. The separate [script compilation report](../benchmarks/script_compile_cpu.json) measures 36–750 repeated observations independently of RNG/model execution; it is not an end-to-end throughput benchmark.

## Precision and reproducibility

Float64 is the compatibility default on CPU and GPU. GPU Sobol PV and all Greeks agreed with DAL in these million-path runs: maximum relative PV error 8e-14, maximum relative Greek error 7.2e-13. Near-zero risks use absolute tolerances in tests. Cross-platform arithmetic need not be bitwise identical.

`dtype="float32"` casts model/path/script arrays; normals are generated in float64 before casting. Block sums use JAX's float32 reduction, then block/device accumulation uses float64; x64 stays enabled. `dtype="auto"` explicitly opts into float32 on GPU and float64 on CPU. For European, scalar Asian and the monthly/weekly barriers here, the largest float32 PV error was about 1e-06, and the largest relative Greek error (excluding autocall) was 0.00013. Short GPU fixtures use PV rtol 2e-5, Greek rtol 5e-3 and atol 2e-4; these are fixture tolerances, not a universal accuracy guarantee.

The autocall repeatedly tests `alive=1` after fuzzy updates. Small path-rounding changes near narrow transitions can be amplified by later conditions. At width .01, float64 produced `d_spot=-3.54332` and `d_vol=-87.20916`; float32 produced approximately `-32.07422` and `+96.57239` while PV differed by less than 3e-7 relatively. Promoting only script evaluation to float64 retained the large risk error: the rounded paths themselves matter. C1 smoothing at the same narrow width also failed to eliminate the error in a diagnostic run.

[The precision stress report](../benchmarks/p4_autocall_precision.json) records widths .01, .1 and 1 on common Sobol paths. Width .1 still has material risk error. Width 1 passes the GPU test tolerance, but changes fuzzy PV from about 125.96025 to 120.00612. Wider smoothing defines a different valuation; use float64 for the original tightly smoothed autocall and validate each product before using float32 Greeks.

Sobol path ids are independent of device count and block size. Normal reduction agrees within floating-point tolerance; deterministic reduction is bitwise invariant to virtual CPU device count with a fixed block size. GPU tests cover all four strategies on the available GPU, different Sobol block sizes, repeated deterministic valuation, bridge on/off, scan/unrolled price and gradient agreement to 1e-14, and explicit CPU placement even when JAX defaults to GPU. Multi-GPU hardware was not tested.

## PRNG choice

| GPU float32 workload | Threefry | RBG |
|---|---:|---:|
| European, price | 2.5 | 2.6 |
| European, greeks | 4.1 | 4.9 |
| Weekly barrier, price | 271.8 | 274.3 |
| Weekly barrier, greeks | 1181.6 | 1186.4 |

These are `mrg32`-named JAX streams, seed 1024; neither implementation reproduces DAL's MRG32 sequence. Tests compare statistics, not DAL path-by-path values. RBG gave no consistent benefit here, so Threefry stays the JAX default; RBG remains explicit through `prng_impl="rbg"`. Raw runs: [Threefry](../benchmarks/p4_gpu_threefry.json), [RBG](../benchmarks/p4_gpu_rbg.json).

JAX documents that native RBG `vmap` can generate a batch from its first key. The engine wraps block `fold_in` and normal generation together with `sequential_vmap`, preserving all block keys under nested batching and GSPMD. Regression tests cover `rbg` and `unsafe_rbg`, prices and all Greeks across four virtual CPU devices and all parallel strategies. See [JAX RNG semantics](https://docs.jax.dev/en/latest/jax.random.html#advanced-rng-configuration). PRNG streams remain independent of device placement with a fixed block size; changing block size changes the stream. RBG is not promised bitwise identical across platforms.

## Reproduce

```bash
uv sync
uv run python benchmarks/bench_suite.py --devices 1 --dal --output cpu1.json
uv run python benchmarks/bench_suite.py --devices 8 --output cpu8.json
uv run python benchmarks/bench_suite.py --devices 8 --parallel auto --output cpu_auto.json
uv run python benchmarks/bench_suite.py --devices 8 --block-size 2048 --output cpu_block.json

# Install GPU wheels in an isolated environment; choose the extra for the driver.
uv venv --python 3.13 /tmp/dal-jax-gpu
uv pip install --python /tmp/dal-jax-gpu/bin/python -e '.[cuda13]' pytest scipy dal-python
XLA_PYTHON_CLIENT_MEM_FRACTION=.35 /tmp/dal-jax-gpu/bin/python benchmarks/bench_suite.py --platform gpu --dtype float64 --repeat 3 --output gpu64.json
XLA_PYTHON_CLIENT_MEM_FRACTION=.35 /tmp/dal-jax-gpu/bin/python benchmarks/bench_suite.py --platform gpu --dtype float32 --repeat 3 --output gpu32.json
XLA_PYTHON_CLIENT_MEM_FRACTION=.35 /tmp/dal-jax-gpu/bin/python benchmarks/bench_suite.py --platform gpu --dtype float32 --rsg mrg32 --prng-impl rbg --repeat 3 --output rbg.json
XLA_PYTHON_CLIENT_MEM_FRACTION=.35 /tmp/dal-jax-gpu/bin/python benchmarks/bench_suite.py --platform gpu --cases autocall --mode greeks --smooth 1 --dtype float32 --block-size 32768 --output wide_autocall.json

uv run pytest
XLA_PYTHON_CLIENT_MEM_FRACTION=.35 /tmp/dal-jax-gpu/bin/python -m pytest tests/gpu --run-gpu
```

The optional [GPU workflow](../.github/workflows/gpu.yml) is manually triggered on a self-hosted Linux runner labelled `gpu`, with a `cuda12`/`cuda13` choice. CPU CI continues on Python 3.13 and 3.14. Consult [JAX's installation guide](https://docs.jax.dev/en/latest/installation.html) for driver and GPU compatibility; the local validation used CUDA 13 wheels.
