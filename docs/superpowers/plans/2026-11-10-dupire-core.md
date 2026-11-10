# Dupire Core Parity Implementation Plan (phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mirror dal-python 2026.11.1's Dupire side — calibration, model data, script risk planners, pullback, quote-space curvature — with both-sides oracle parity, building on the merged PR-#9 risk machinery.

**Architecture:** New `src/dal_jax/models/dupire.py` (IVS bases, DupireRiskInputs, FD calibration → existing LocalVolSurface, model data) + planner/pullback/curvature entry points alongside the risk machinery; api.py shims; oracle tests.

**Spec:** `docs/superpowers/specs/2026-11-10-dupire-core-design.md` (binding; carries the verified probe semantics and reference values).

---

### Task 1: IVS bases + DupireRiskInputs + calibration snapshot parity

Probes first (ctor kwarg matrices, validation messages), TDD, then: BS/Merton IVS bases; callable-IVS acceptance (documented ergonomic divergence); `DupireRiskInputs_` dataclass + validation; the FD Dupire calibration (`DupireCentral1e-4FixedBaseBand/v1` stencil from /tmp/dalsrc — read the C++ before implementing) onto the densified inclusion grid; snapshot dataclass with `.spots/.times/.vols/.algorithm/.matches`. Oracle: snapshot parity (grid + vols) for the spec's fixture (strikes [75,105,135] × maturities [0.4,1.2], inclusion [60,100,140]×[0.5,1.0]) across all three base kinds.

### Task 2: DupireModelData + engine pricing/risk parity

`DupireModelData_New(cal, index, ccy, discount, max_step)` → a dal-jax model on LocalVolSurface plumbed into the standard engine. Oracle: `MonteCarlo_ValueWithSettings` PV parity (anchor 105.040477399776 @ 4096) and `MonteCarlo_ValueWithRisk` 21-input axis + jacobian parity (18 `model:N` surface nodes + spot + div + rate).

### Task 3: Script risk planners + pullback

`DupireScriptRiskRequest_`/`DupireQuoteBinding_`/`CalibrationRiskRequest_` mirrors; `DupireScriptRiskPlan_New`/`DupireScriptRiskResult_New` chains with method strings (`NativeAADThenNativeAADCalibrationVJP`, retrained, expired), quote-ordinal selection (`quote:N`), `plan.quote_plan.selected_ordinals`, required vs complete axis, report_factors scaling (anchor jacobian [[22.257, 3.0437]]); `DupireParameterAdjoints_FromRisk` → `DupireQuoteRisk_New` (unit `decimal-vol`) and `DupireScriptQuoteRisk_New` with additive direct adjoints.

### Task 4: Quote-space curvature

`BumpOverAADRequest_` directions = one column per flattened quote_spreads entry (detached copies); bump-recalibrate-revalue curvature, method `BumpOverRecalibratedNativeDupireMonteCarloAAD`, hessian in quote space; both-sides parity.

### Task 5: Closeout

Full targeted suites + full suite; gates ×4; backend-init False; CHANGELOG (incl. callable-IVS divergence note); push + PR (base master).
