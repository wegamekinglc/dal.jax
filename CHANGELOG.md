# Changelog

## Unreleased

- Add the request-driven risk API from DAL 2026.11.1: `MonteCarlo_ValueWithRisk`, `MonteCarlo_ValueWithJacobianRisk` and `MonteCarlo_ValueWithWeightedRisk` with `RiskRequest_`-style coordinate selection over the model-parameter and script-constant axes, `Product_Get_RiskOutputs` slots derived from the scalar variable table, native-matching provenance and legacy `PV`/`d_*` values, and pre-binding EXERCISE/fully-expired guards for the Jacobian and weighted estimators.
- Add an exact `PseudoRSG` (MRG32k3a) compatibility stream with native antithetic uniforms and precise normals (`PseudoRSG_New`, `PseudoRSG_Get_Uniform`, `PseudoRSG_Get_Normal`).
- Wire `normal_precision='Precise'` to the precise Sobol inversion (pricing, LSMC training and RQMC replays share one composition rule); make `lsmc_training_memory_mb` force the checkpointed training scan with unchanged numbers.
- Add the `BlackScholesMonteCarlo` plan API (`Plan_New`, `Get_Risk`, `Get_Curvature`) with script constants joining the parameter labels, `first_path` as a Sobol point-ID offset, curvature via bump revaluation of the AAD gradient, and `SegmentedMonteCarloSettings_` execution settings. `GSRMonteCarloPrice_` stays unmirrored: native Python never constructs it (calibration-internal). Known gaps documented in the code: pseudo-RSG `first_path` has no effect (streams are not bit-compatible), `segment_steps`/budget fields are validate-and-echo, native AAD-tape telemetry is reported as `None`, and expired products report method `NativeAAD` where native says `Expired`.
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
