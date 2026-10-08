# Configuration reference

Native settings are frozen dataclasses. Input sequences are copied into tuples; changing caller lists later cannot change the configuration. Use `dataclasses.replace` to construct a revised value, then build a new engine.

## Valuation and product settings

| Type / field | Default | Meaning |
| --- | --- | --- |
| `ValuationContext.evaluation_date` | Required | Date used to partition past and future events |
| `ValuationContext.fixings` | Empty `FixingSnapshot` | Explicit historical market data; no global fallback |
| `ValuationContext.today_fixing_policy` | `TodayFixingPolicy.MODEL` | `REQUIREHISTORICAL` also treats today as historical |
| `ValuationSettings.evaluation_date` | `None` | Legacy date fallback during context resolution |
| `ValuationSettings.fixings` | `None` | Legacy fixing fallback during context resolution |
| `ValuationSettings.today_fixing_policy` | `TodayFixingPolicy.MODEL` | Same policy as the complete context |
| `ScriptProductSettings.default_index` | `""` | Default named index for `SPOT()` |
| `ScriptProductSettings.regression_features` | `()` | LSMC equity/rate indices or scalar script state |

## Monte Carlo settings

All fields below belong to `MonteCarloSettings`.

| Field | Default | Accepted values and behavior |
| --- | --- | --- |
| `rsg` | `"sobol"` | `sobol`, `mrg32`, `irn`; pseudo-random names match DAL statistically, not by stream |
| `use_bb` | `False` | Apply a factor-aware Brownian bridge when supported |
| `enable_aad` | `False` | Report risks and enable fuzzy script conditions |
| `smooth` | `0.01` | Finite positive fallback smoothing width; scripts can specify condition widths |
| `smoothing_kernel` | `"dal"` | `dal` linear kernels or `smoothstep` cubic C1 kernels |
| `scan_group_threshold` | `4` | Minimum adjacent equal event count for grouping; `0` disables it |
| `inverse_normal` | `"acklam"` | `acklam`, `acklam_polish`, `acklam_polish_precise`, `ndtri` |
| `sobol_shift_key` | `None` | Nonnegative 64-bit SplitMix64 digital-shift key, Sobol only |
| `seed` | `1024` | Integer pseudo-random seed |
| `prng_impl` | `None` | JAX implementation selection; `None` uses its configured default |
| `block_size` | `"auto"` | Positive integer; auto uses 8192 on CPU and a memory-based GPU estimate |
| `parallel` | `"shard_map"` | `shard_map`, `auto` (GSPMD), `pmap`, `none` |
| `platform` | `"auto"` | `auto`, `cpu`, `gpu`; auto uses JAX's default backend |
| `devices` | `None` | Explicit nonempty device tuple; otherwise resolve all local selected-backend devices |
| `dtype` | `"float64"` | `float64`, `float32`, `auto`; auto uses float32 on GPU and float64 on CPU |
| `deterministic_reduction` | `False` | Ordered block values/Jacobians independent of device count; custom reverse mode |
| `block_bucketing` | `False` | Round ordinary-pricer block counts to powers of two to reuse compilation |
| `checkpoint` | `True` | Rematerialize block and grouped-event intermediates during reverse mode |
| `lsmc_basis_degree` | `3` | Scalar 1–8; multiple features 1–3 |
| `lsmc_training_paths` | `None` | Positive count; default equals paths per pricing replica |
| `lsmc_validation_paths` | `None` | Nonnegative held-out count; `None` or zero disables degree selection |
| `lsmc_rqmc_replicates` | `None` | At least two randomized Sobol replicas, or one unshifted/explicitly shifted interval by default |
| `lsmc_training_seed` | `None` | Nonnegative 31-bit seed; requires RQMC, defaults to zero when enabled |
| `lsmc_pricing_seed` | `None` | Nonnegative 31-bit seed; requires RQMC, defaults to zero when enabled |
| `lsmc_policy_risk_mode` | `"Frozen"` | `Frozen` or `RetrainedBump`; retrained correction requires AAD |
| `lsmc_policy_bump_relative` | `0.001` | Finite value in `(0, 0.1]`; bump fraction of `max(1, abs(parameter))` |

Training, validation and all pricing replicas together must fit within `2**32 - 1` Sobol points. LSMC supports Sobol only. Block padding is masked out of pricing; record collection returns actual paths in global order.

## GSRSLV settings

| `GSRSLVSettings` field | Default | Meaning |
| --- | --- | --- |
| `max_step` | `1 / 52` | Maximum ACT/365 simulation interval |
| `kappa` | `1.0` | Variance mean reversion towards the unit long-run level |
| `vol_of_vol` | `0.5` | Volatility of the square-root variance process |
| `variance_correlations` | `()` | Correlations between rate factors and variance noise |

The DAL-compatible `GSRSLVSettings_()` constructor remains a mutable input builder. `GSRSLVModelData_New` snapshots it into native frozen settings; changing the builder does not reconfigure an existing model.

## Process configuration

Importing `dal_jax` enables JAX x64 to retain the existing numerical default. `config.configure(num_cpu_devices=..., prng_impl=..., compilation_cache_dir=...)` configures JAX's process runtime; call it during startup, before the first JAX operation. These settings are shared by the third-party runtime and are not request-scoped. A new process is required when changing initialized device topology.

For memory, precision and reproducibility tradeoffs, use [the performance guide](performance.md). For model inputs and constraints, use [the rates and LSMC guide](p6-p7.md) and [the P5 guide](p5.md).
