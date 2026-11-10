# Dupire core parity with dal-python 2026.11.1 — design (phase 1 of 2)

Date: 2026-11-10
Status: approved scope direction (phasing + ⑤ deferral decided 2026-11-09/10);
this spec covers phase 1. Phase 2 (MarketIv: containers, ConstrainedSSVIFit,
refit risk, legacy bridge) is a separate spec.
Branch: `feature/dupire-core` (to be created off master; depends only on the
merged PR #9 risk machinery, not on PR #10).

## Probe basis

The full 56-name domain is Python-reachable with zero gates (see the
dupire-probe report, 2026-11-09; scripts preserved under /tmp/dupire_probe
at the time of writing). Phase 1 mirrors the Dupire side only — it needs no
MarketIv fitted surface: base IVS kinds BS / Merton / Python-callable cover
calibration inputs.

## Verified native semantics to mirror (from the probe)

- `DupireRiskInputs_(quote_strikes, quote_maturities, quote_spreads (DoubleMatrix
  strikes×maturities), inclusion_spots, max_spot_spacing, inclusion_times,
  max_time_spacing, ...)`.
- `DupireCalibration_New(base_ivs, inputs, name=...)` → snapshot with
  `.spots/.times/.vols` (densified inclusion grid; vols DoubleMatrix spot-major),
  `.spot/.rate/.dividend_yield`, `.algorithm`
  (`DupireCentral1e-4FixedBaseBand/v1`), `.surface` (opaque), `.matches()`.
  Base IVS accepts `BSModelData_`, `MertonIVS_(spot, vol, intensity,
  average_jump, jump_std)`, and Python subclasses of `dal.IVS_` overriding
  `implied_vol(strike, maturity)` (a trampoline dal-jax cannot and need not
  replicate 1:1 — our mirror accepts callables natively, an intentional
  ergonomic win documented in CHANGELOG).
- `DupireModelData_New(cal, index_name, ccy, discount, max_step)` → a
  `ModelData_` that prices through the standard script MC engine:
  `MonteCarlo_ValueWithSettings` PV parity and `MonteCarlo_ValueWithRisk`
  AAD jacobian over the 21-input axis (18 surface nodes + spot + div + rate).
- Script risk planners: `DupireScriptRiskRequest_(num_paths, quotes=
  CalibrationRiskRequest_(inputs=["quote:N"...], report_factors=...),
  direct_bindings=[DupireQuoteBinding_(constant_ordinal, quote_id)], valuation,
  simulation)` → `DupireScriptRiskPlan_New(product, model, calibration,
  component, request)` → `DupireScriptRiskResult_New(plan)`. Method string
  `NativeAADThenNativeAADCalibrationVJP`; quote ids are ordinals into the
  flattened quote grid; `plan.quote_plan.selected_ordinals`;
  `plan.required_input_axis` (surface nodes + bound constants; spot/div/rate
  excluded from required, present in complete); LSMC retrained mode method
  string; expired mode (`ExpiredThenNativeAADCalibrationVJP`, PV 0).
- Pullback: `DupireParameterAdjoints_FromRisk(risk_result, cal, component)` →
  `DupireQuoteRisk_New(cal, seeds)` (method `NativeAADCalibrationVJP`, unit
  `decimal-vol`) and `DupireScriptQuoteRisk_New(..., direct=
  DupireDirectQuoteAdjoints_(cal, matrix))` with additive direct adjoints.
- Curvature: `BumpOverAADRequest_(directions, steps)` — directions have ONE
  COLUMN PER FLATTENED quote_spreads entry (strikes×maturities, not surface
  nodes); `.directions` returns a detached copy; result method
  `BumpOverRecalibratedNativeDupireMonteCarloAAD`, hessian in quote space.

## Phase 1 scope

New module `src/dal_jax/models/dupire.py` (calibration + model data) with
engine integration, plus risk-planner/pullback/curvature entry points in
`src/dal_jax/mc/` per the codebase's module conventions; api.py shims +
aliases; oracle tests `tests/oracle/test_dal_dupire.py`.

1. **Dupire calibration in JAX**: FD Dupire local-vol from an IVS callable
   onto the densified inclusion grid → the existing `LocalVolSurface(spots,
   times, vols)` (layout already identical, spot-major). Target: snapshot
   `.vols` parity vs native within the probe-reproducible tolerance (native
   algorithm string pinned; central-difference parameters per C++
   `DupireCentral1e-4FixedBaseBand/v1` — read /tmp/dalsrc for the exact FD
   stencil if probes show nontrivial differences).
2. **Model + pricing parity**: `DupireModelData_New` → dal-jax model
   plumbed into the standard engine; `MonteCarlo_ValueWithSettings` /
   `ValueWithRisk` parity on the probe's fixture (PV 105.04 @ 4096 paths;
   21-input axis incl. `model:N` surface-node labels).
3. **Risk planners / pullback / curvature**: mirror the request/plan/result
   chains, method strings, quote-ordinal selection, report_factors scaling,
   direct adjoints, retrained/expired modes, and the quote-space curvature —
   each both-sides oracle-pinned.
4. **Out of phase 1**: all `MarketIv*` names, the SSVI fit engine, refit
   risk, `MarketIvSurfaceAsLegacyIVS` bridge (phase 2 spec).

## Errors and validation

Probe every constructor/validation message; register new error codes
following the established pattern (`InvalidSetting`-style + `_SCRIPT_CODES`).

## Testing

Both-sides oracle parity per behavior (the PR-#9/#10 discipline); embedded
probe anchors (PV 105.040477399776 @ 4096; risk values [104.361...];
jacobian [[22.257, 3.0437]] for the quote-subset case).

## Documentation

CHANGELOG entry including the intentional callable-IVS ergonomic divergence
and the phase-2 pointer.
