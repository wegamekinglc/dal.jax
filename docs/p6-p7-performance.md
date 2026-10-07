# LSMC and rate-model performance (P6/P7)

Measured on 2026-10-08 with float64, JAX/jaxlib 0.11.2 and Python 3.13.9. The host is an Intel Core i9-13900HX under WSL2 with 32 logical processors; the GPU is an RTX 4060 Laptop, 8 GiB, CUDA 13 wheels, driver 595.79. DAL uses 32 CPU threads and source commit `4feabe89b105e0a3883fd4e2d74a2d70d618d8c7`. Timed workload processes ran sequentially; every JAX call was synchronized. Tables use the median of three warm runs in milliseconds.

## Million-path Bermudan

The four-date put has spot 100, volatility .15, rate .05, dividend .03 and strike 100. It uses 65,536 training paths followed by 1,048,576 disjoint pricing paths, degree 3 and Brownian bridge. Exact price and fuzzy price plus all five model/script risks are separate rows. Blocks contain 8,192 paths. No validation paths or randomized replicas are used here.

| Mode | Backend | Training | Fixed-policy pricing | Training + pricing | DAL training + pricing |
|---|---|---:|---:|---:|---:|
| Exact price | CPU 1 | 31.6 | 96.8 | 130.8 | 19.1 |
| Exact price | CPU 4 | 49.3 | 25.6 | 74.8 | 17.0 |
| Exact price | GPU 1 | 42.2 | 23.8 | 71.6 | 16.6 |
| Price + risks | CPU 1 | 32.0 | 337.6 | 380.4 | 122.2 |
| Price + risks | CPU 4 | 48.6 | 154.6 | 217.4 | 111.8 |
| Price + risks | GPU 1 | 40.6 | 98.1 | 153.1 | 113.5 |

Fixed-policy pricing is the compiled `pricer` executable with no host training. Complete valuation is `engine.value`, including fresh training, host regression, pricing, validation and scalar result conversion. DAL's API also trains on every call. Preparation is outside both timers. Training and pricing medians need not add to the complete-call median because they are separately sampled.

Training's first call took .86–1.14 seconds in the exact-price rows, including tracing/compilation and host fitting. Pricing compilation took .28–.52 seconds for exact price and 1.07–1.65 seconds for price plus risks. The Greek row can reuse training compilation from the preceding exact row; these are first-call measurements in a shared process, not independent cold starts. Raw reports retain all first calls and warm samples: [CPU 1](../benchmarks/p6_cpu_1.json), [CPU 4](../benchmarks/p6_cpu_4.json), [GPU](../benchmarks/p6_gpu_64.json).

The exact PV is `6.49669058408335`; fuzzy PV is `6.49664325864306`. The warm fixed-policy GPU risk calculation is faster than the DAL complete call, but fresh training makes the JAX complete call slower. Scalar regression uses device moments and host small-system solves; launch, transfer and host overhead matter at these training sizes. Four CPU devices improve pricing throughput but do not improve the host regression phase. These measurements do not establish a general speed advantage.

## Rate and hybrid examples

Examples 15 and 16 use 65,536 pricing paths, float64, Brownian bridge and 1,024-path blocks. Every row includes all active model/script risks. The Gaussian examples price a bond, caplet and five-year swap together, with an explicit three-month projection curve. The rate Bermudan trains on 4,096 paths. The SLV cases use a two-factor Gaussian kernel, a leverage surface, stochastic variance and a maximum step of .25 years; the equity hybrid includes named cross-correlations. Full parameters and grid/factor details are in the scripts and JSON.

| Product | CPU 4 | GPU 1 | DAL CPU reference* |
|---|---:|---:|---:|
| Single-factor GSR | 20.9 | 14.4 | 21.3 |
| Two-factor GSR | 14.2 | 23.0 | 27.6 |
| Rate Bermudan, fixed policy | 15.0 | 26.7 | 15.7 |
| Standalone GSRSLV | 29.7 | 60.0 | 19.0 |
| BS equity + GSRSLV | 46.4 | 94.5 | 24.9 |
| Local-vol equity + GSRSLV | 55.3 | 106.5 | 34.3 |

\* DAL reference values shown are from the matching GPU example process; CPU example JSON includes its separately measured DAL calls. Rate Bermudan JAX times exclude training, while DAL times include it. Other rows exclude preparation but include the complete compiled price/gradient calculation.

Pricing compilation ranged from 1.92–2.89 seconds on CPU and 2.97–4.92 seconds on GPU. The consumer GPU has limited float64 throughput and these short-block SLV kernels are slower than DAL. No performance claim is made for float32 risk; separate GPU tests validate three hybrid fixtures with relaxed precision tolerances.

Raw reports: [GSR CPU](../examples/results/15_cpu.json), [GSR GPU](../examples/results/15_gpu.json), [SLV/hybrid CPU](../examples/results/16_cpu.json), [SLV/hybrid GPU](../examples/results/16_gpu.json).

## Validation, replicas and policy risk

Example 13 uses 65,536 paths for both training and pricing. Its JSON reports regression coefficients, first-exercise rates, a fixed-policy spot ladder, training and complete-call timings. Example 14 uses 4,096 training paths, 1,024 held-out paths and three independent 65,536-path digital-shift pricing replicas. Its native diagnostic comparisons check each replica mean, with uncertainty conditional on one trained policy.

| Workload | CPU 4 | GPU 1 | DAL CPU reference* |
|---|---:|---:|---:|
| Bermudan fixed-policy price + risks | 16.5 | 20.7 | 21.1 |
| Validated RQMC fixed-policy price + risks | 23.3 | 8.8 | 24.0 |
| RetrainedBump complete call, 1,024 train / 4,096 price | 37.7 | 111.9 | 11.3 |

The retraining comparison has different path counts from the RQMC row. It includes batched JAX policy fitting, base-input repricing and host constraint checks. Its first complete call took 3.70 seconds on CPU and 6.49 seconds on GPU; repeated calls reuse the executable. Native policy risks agree, including one-sided corrections at constrained zero-volatility parameters in the oracle tests.

Raw reports: [Bermudan CPU](../examples/results/13_cpu.json), [Bermudan GPU](../examples/results/13_gpu.json), [RQMC/retraining CPU](../examples/results/14_cpu.json), [RQMC/retraining GPU](../examples/results/14_gpu.json). Across these eleven benchmark/example reports the maximum PV relative difference was `4.2e-15`; the largest absolute difference in any reported quantity was `2.8e-13`. Near-zero risks are checked with an absolute tolerance.

## Reproduce

Build the native reference in an isolated environment, as described in [the examples guide](../examples/README.md). Do not resynchronize the environment with the older published `dal-python` after installing the source oracle.

```bash
/tmp/dal-jax-native/bin/python benchmarks/bench_lsmc.py --devices 1 --output cpu1.json
/tmp/dal-jax-native/bin/python benchmarks/bench_lsmc.py --devices 4 --output cpu4.json
XLA_PYTHON_CLIENT_MEM_FRACTION=.35 /tmp/dal-jax-gpu/bin/python benchmarks/bench_lsmc.py --platform gpu --devices 1 --output gpu.json

for example in examples/1[3-6]_*.py; do
  /tmp/dal-jax-native/bin/python "$example" --paths 65536 --devices 4 --repeat 3 --output "/tmp/$(basename "$example" .py)-cpu.json"
done
```

The GPU environment needs the matching CUDA extra and the same source oracle. Run examples with `--platform gpu --devices 1`; use separate processes for device-count changes. GPU processes here used allocator fraction .35. Only one physical GPU was tested. Laptop clocks, WSL scheduling and other host activity affect timing; raw samples are retained to make that variability visible.
