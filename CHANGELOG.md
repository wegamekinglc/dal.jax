# Changelog

## Unreleased

- Upgrade to dal-python 2026.11.1 and JAX 0.12, and track uv.lock in git so every workflow resolves reproducible dependency sets instead of silently following new PyPI releases.
- Match the 2026.11.1 DAL diagnostics: report `normal_precision`, `lsmc_checkpoint_block_events` and `lsmc_training_memory_mb` (with native validation rules) and mirror `compiled` in the simulation settings explanation.
- Parse variadic `MIN`/`MAX` as left-folded binary chains (`MIN(a, b, c)` becomes `min(min(a, b), c)`), matching the 2026.11.1 DAL AST shape in `Product_Describe` dumps.
- Compare LSMC training coefficients only for full-rank fits: rank-deficient regressions are unique modulo the null space, so the oracle test pins degrees, ranks and values there.
- Run the native-oracle CI and GPU jobs against the pinned dal-python wheel from the lockfile; the wheel now ships the full oracle API, and building the C++ source on the runner drifted numerically with the Ubuntu 26.04 toolchain.
- Freeze engine configuration and snapshot sequence inputs; expose explicit valuation contexts and owned sessions while deprecating legacy global setters.
- Return immutable LSMC training/valuation results, move caches outside transformed kernels, and add per-engine cache clearing.
- Batch GSR, GSRSLV and Hybrid coefficient preparation; use a reverse device-training scan and CPU ordered Welford scan while retaining measured GPU fallbacks.
- Standardize Python with Ruff and CI checks; consolidate feature explanations, settings, numerical contracts and migration guidance in user documentation.

## 0.1.0a1

- Pure Python/JAX Monte Carlo valuation with CPU and CUDA extras, float64 defaults, blocked execution and four parallel strategies.
- DAL-compatible scalar/vector scripts, schedules, historical fixings, delayed payments, fuzzy risks and JAX differentiation.
- Black–Scholes, correlated equities, local volatility, single/multi-factor GSR, GSRSLV and named-factor domestic hybrids.
- Three-phase LSMC with normalized regression, rank guards, validation-based degree selection, independent randomized Sobol replicas and frozen/retrained-policy risks.
- Preparation/simulation diagnostics, sixteen executable DAL comparison examples and CPU/GPU validation.
