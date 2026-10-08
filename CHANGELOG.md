# Changelog

## Unreleased

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
