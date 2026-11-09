# Portfolio risk API parity with dal-python 2026.11.1 — design

Date: 2026-11-09
Status: approved (sub-project ② of the continuation program; ① polish batch runs in parallel)
Branch: `feature/polish-and-instruments`

## Goal

Mirror the Python-reachable portfolio valuation surface of dal-python 2026.11.1:
`ScriptPortfolio_New` plus `PortfolioMonteCarlo_ValueWithJacobianRisk` and
`PortfolioMonteCarlo_ValueWithWeightedRisk`, with the portfolio-level
request/result dataclasses. This builds directly on the per-product risk
machinery delivered in PR #9.

## Scope decisions (probe-verified)

- **The six IR instrument factories** (`Deposit/Fra/Swap/BasisSwap/PeriodForwardOis/EuropeanRateOption
  ScriptInstrument_New`) are **not mirrored**: they consume native `YCInstrument_`
  handles that cannot be constructed from Python (probed: `Read<>` rejects plain
  objects/strings), exactly the `GSRMonteCarloPrice_` precedent. Documented in
  CHANGELOG instead.
- **`ScriptInstrumentDefinition_*`** is deferred: the spec constructor requires a
  typed `source` structure tied to the calibration import workflow; revisit if a
  later sub-project needs it.
- `CalibrationInstrument_BindParameters` and the `PreparedCalibrationInstrument_`
  telemetry stay out (calibration scope, per handoff).

## Verified native semantics (probed against the wheel)

- `ScriptPortfolio_New(trade_ids, products, modelData)` — `modelData` is a
  **per-trade list of models** (one entry per trade; `ScriptPortfolioData_`
  exposes `model_owners`, here `[0, 1]`). Validation probed so far:
  `InvalidScriptPortfolio: ScriptPortfolio_New; modelData; type=ModelData_; expected list or tuple, or None`
  (single model object rejected); passing `None` also rejected with
  `type=NoneType, value=None; expected list or tuple, or None` (native quirk —
  keep the message verbatim); unequal lengths →
  `InvalidScriptPortfolio: trade_ids, products and modelData must have equal nonzero lengths`.
  The result exposes `name` (empty default), `trade_ids`, `model_owners`.
  **Identity dedup (as built, probed):** `model_owners` deduplicates by object
  *identity*, not equality — the same object passed twice gives `[0, 0]`,
  equal-but-distinct models give `[0, 1]`, and a model reused after a gap
  (`[m1, m3, m1]`) keeps its first owner (`[0, 1, 0]`).
- `PortfolioMonteCarlo_ValueWithJacobianRisk(portfolio, num_path, *, request,
  valuation)` returns a result with: `values` (one PV per trade),
  `complete_input_axis` / `input_axis` (concatenated per-trade RiskCoordinate
  lists), `complete_output_axis` / `output_axis` (per-trade RiskOutputCoordinate
  lists), `jacobian` / `reported_jacobian` (DoubleMatrix), `execution`
  (PortfolioRiskExecution_), `provenance` (PortfolioRiskProvenance_).
- Requests: `PortfolioJacobianRiskRequest_`, `PortfolioWeightedRiskRequest_`
  (fields probed at implementation time — see plan).

## Design

New module `src/dal_jax/mc/portfolio.py`; shims + aliases in `api.py`;
oracle tests in `tests/oracle/test_dal_portfolio.py`.

1. `ScriptPortfolioData` dataclass: `trade_ids`, `products`, `models`,
   `model_owners` (owner index per trade; semantics — dedup of identical
   models? — probed and pinned at implementation), `name=""`.
2. `ScriptPortfolio_New(trade_ids, products, model_data)`: length/type
   validation with native-exact messages; builds the dataclass.
3. `portfolio_jacobian_result(portfolio, n_paths, request, valuation)`:
   one engine run per trade via the existing PR-#9 machinery (per-trade
   `jacobian_risk_result`-equivalent), assembled into the cross-trade
   jacobian (cross-model block semantics probed natively first — likely
   block-diagonal since trades value independently), concatenated axes
   (label collisions across trades probed: how does native disambiguate
   `spot` from trade 1 vs trade 2 — prefixes? ids?), execution/provenance
   with native-matching fields.
4. `PortfolioMonteCarlo_ValueWithWeightedRisk` mirrors the weighted variant
   (weights vs trades/outputs per native probe).
5. Dataclasses mirror every native attribute exactly (the PR-#9 discipline:
   probe `dir()`, match field-for-field, pin with a live attr-set equality
   oracle test).

## Testing

Both-sides oracle tests per behavior (values, axes, jacobian, validation
errors, request filtering), same discipline as PR #9. Unit tests for
dal_jax-only edges.

## Documentation

CHANGELOG entry including the instrument-factory/definition exclusion
rationale.

## Implementation notes (as built, 2026-11-09)

The trade-axis guard has two distinct causes, both wrapped in the
trade-context message and raised before any selector validation: a
product with neither PAYS nor EXERCISE fails the guard's own check
(`cause=product must contain a PAYS or EXERCISE payoff`), while
EXERCISE-without-PAYS passes it and is rejected by the delegated
risk-output axis (`cause=InvalidWeightedRiskRequest: indexed scalar
default receiver is required; output=payoff`, verbatim). The
exercise/expired guard message omits native's
`InvalidPortfolioPreparation: trade=<id>; cause=...` wrapper and raises
the bare inner `UnsupportedPortfolioRisk: ...` code, consistent with the
PR-#9 convention of matching native's final error line.

A group's `prefix_reversals` counts one tape reversal per executed
replay batch: native's batch decomposition scales with its thread count
(`portfoliobatch.cpp` reverses once per batch — 32 threads split 64
paths into 32 batches locally, a 4-vCPU runner into 4), so the count is
native-thread-scaled telemetry; dal-jax reports its own block layout's
batch count and the oracle assertion is structural, like the tape byte
peaks and the plan API's lanes. `suffix_reversals` (one per path) is
machine-independent and stays exactly asserted.
