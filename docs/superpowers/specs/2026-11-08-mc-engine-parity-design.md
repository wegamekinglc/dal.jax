# MC engine parity with dal-python 2026.11.1 — design

Date: 2026-11-08
Status: approved (方案一: compatibility shims over the existing engine)
Branch: `feature/mc-engine-parity` (stacked on `fix/pin-dal-python-oracle`)

## Goal

Close the Monte Carlo engine API gap between dal-jax and the pinned dal-python
2026.11.1 oracle, in scope **A + B** as agreed:

- A (script MC engine core): request-driven risk API, `PseudoRSG_*`,
  `normal_precision` wiring, LSMC memory-field semantics.
- B (adjacent engine features): `BlackScholesMonteCarlo*` plan API with
  `SegmentedMonteCarloSettings_` execution settings.

Out of scope: `GSRMonteCarloPrice_` (no Python constructor; its only consumer
`PriceGSRSLVEuropeanOptions` is calibration machinery, which the handoff doc
declares out of scope), XCCY clusters, script instrument library, rate
quote-risk provenance, calibration, Dupire/MarketIv.

## Verified native semantics (probed against the 2026.11.1 wheel)

- `MonteCarlo_ValueWithRisk(product, model, n, request=None, valuation=,
  simulation=)` returns `RiskResult_` with: `values` (PV per payoff slot),
  `output_ids`/`output_axis`, `input_axis` and `complete_input_axis` (one
  `RiskCoordinate_` per model input: family='model', id='model:N', label in
  spot/vol/rate/div order, ordinal, native_unit='model-coordinate',
  physical_unit, report_scale, value), `jacobian` (outputs x inputs; the legacy
  `d_*` values are its rows), `reported_jacobian` (request-filtered),
  `legacy_values` ({PV, d_*}), and `provenance` (engine, method e.g.
  'NativeAAD', normalization='mean', calibration='fixed', evaluation_date,
  execution snapshot).
- `WeightedRiskRequest_(weights=[...])`: weight count must equal the selected
  output count; result adds `weighted_value = sum(w_i * value_i)`,
  `component_means`, `output_axis`, `weights`.
- `JacobianRiskRequest_` adds `max_block_width=1` plus recording/scratch
  capacity budgets.
- `BumpOverAADRequest_(directions, steps, numeric_payload_budget_bytes=None,
  recording_capacity_budget_bytes=None)`.
- `Product_Get_RiskOutputs(product)` -> list of `RiskOutputCoordinate_`
  ({id: 'payoff', label: <payoff var>, slot: N}).
- `PseudoRSG_New(seed, ndim=1)`; `Get_Uniform/Get_Normal(rsg, n)` return
  n x ndim `DoubleMatrix` and advance the stream by n rows per call.
- `BlackScholesMonteCarloPlan_New(script_product, valuation=None,
  smoothing=0.01)` accepts script products; the plan exposes contract_dates,
  contract_events, observations, parameter_labels (['spot','vol','rate','div']),
  script_constants, smoothing. `Get_Risk(plan, point, n, settings=None)` takes
  the point as a plain sequence in parameter_labels order and returns
  {value, gradient, parameter_labels, point}; `Get_Curvature(plan, point, n,
  bumps: BumpOverAADRequest_, settings=None)`. Execution settings are
  `SegmentedMonteCarloSettings_` (rsg, use_bb, normal_precision, scramble_key,
  first_path=0, segment_steps=64, recording/checkpoint capacity budgets).
- `normalPrecision` in the C++ engine only distinguishes
  `precise = (value == "Precise")` when constructing every RSG; "Fast" is
  accepted by validation but behaviourally equal to "Default".

## Design

All new names are dal-python-compatible shims in `dal_jax/api.py`, backed by a
new `src/dal_jax/mc/risk.py` and a new `src/dal_jax/mc/plan.py`. The existing
engine core is not restructured; wiring changes are local to RSG construction
and the LSMC training scan.

### 1. Request-driven risk API (`mc/risk.py`)

Dataclasses mirroring the native attributes field-for-field:
`RiskCoordinate_`, `RiskOutputCoordinate_`, `RiskRequest_`,
`JacobianRiskRequest_`, `WeightedRiskRequest_`, `BumpOverAADRequest_`,
`RiskResult_`, `JacobianRiskResult_`, `WeightedRiskResult_`,
`RiskResultProvenance_` (simplified execution snapshot: method, block width,
device count).

- `Product_Get_RiskOutputs(product)` derives payoff slots from the prepared
  product's payoff structure (one coordinate per `PAYS`/payoff variable,
  id='payoff', label=variable name, slot=index).
- `MonteCarlo_ValueWithRisk/WithJacobianRisk/WithWeightedRisk` run the
  existing engine with AAD enabled (the current `MonteCarlo_Value(
  enable_aad=True)` path already computes the full jacobian), then:
  - build `input_axis` from the model's existing risk labels and parameter
    values (spot/vol/rate/div for BS and CorrelatedBS per-factor, curve
    parameters for GSR family as already exposed by the legacy `d_*` keys);
  - honour `request.inputs`/`request.outputs` (label or slot selectors) by
    slicing axes, jacobian and values into `reported_jacobian`;
  - keep `legacy_values` for backward compatibility;
  - validate weights exactly like native (count == selected outputs) and add
    `weighted_value`/`component_means`;
  - record capacity/budget request fields in provenance (accepted, not
    enforced — XLA manages memory).
- Errors: unknown labels/slots raise `InvalidSetting`-style `DalError` with the
  native message prefix `InvalidRiskRequest`.

### 2. `PseudoRSG_*` (`random/prng.py` extension)

`PseudoRSG_New(seed, ndim=1)` returns a stateful wrapper over the existing
MRG32 stream; `PseudoRSG_Get_Uniform/Get_Normal(rsg, n)` return `n x ndim`
numpy arrays and advance the stream. Parity with the native stream is an
oracle test; if the seeds differ structurally, align the seed mapping first.

### 3. `normal_precision` wiring (`mc/engine.py`)

At RSG construction: `precise = settings.normal_precision == "Precise" or
settings.inverse_normal == "acklam_polish_precise"`; polish resolves
analogously; `ndtri` stays exact. "Default" and "Fast" remain behaviourally
equal, matching native. The explain report already echoes the field.

### 4. LSMC memory fields (`mc/lsmc.py`)

When `lsmc_training_memory_mb` is set the training scan forces the checkpointed
(rematerialized) variant regardless of the generic `checkpoint` flag; numbers
are unchanged (native treats both fields as memory-only knobs). The resolved
values are reported through the existing simulation diagnostics.

### 5. `BlackScholesMonteCarlo` plan API (`mc/plan.py`)

- `BlackScholesMonteCarloPlan_New(product, valuation=None, smoothing=0.01)`
  prepares the script once and exposes the native plan attributes (dates,
  events, observations, parameter_labels, script constants, smoothing).
- `BlackScholesMonteCarlo_Get_Risk(plan, point, num_path, settings=None)`
  builds a BS model from the point sequence, reuses the jitted pricer and
  `jax.grad` for the gradient; returns value/gradient/parameter_labels/point.
- `BlackScholesMonteCarlo_Get_Curvature(plan, point, num_path, bumps,
  settings=None)` evaluates bump revaluations over `BumpOverAADRequest_`
  directions/steps via the same jitted pricer (second differences).
- `SegmentedMonteCarloSettings_` dataclass with the native fields;
  `segment_steps` maps to the existing block execution, `first_path` offsets
  the random stream, capacity budgets are accepted and recorded.

## Testing

TDD per feature with oracle parity against the pinned wheel
(`tests/oracle/test_dal_risk_api.py`, `test_dal_pseudo_rsg.py`,
`test_dal_bs_plan.py` plus unit tests beside the sources):

- risk API: axis coordinates, jacobian values, request filtering, weights
  validation, provenance fields, `Product_Get_RiskOutputs` slots;
- PseudoRSG: streaming advance and uniform/normal parity;
- normal_precision: precise inversion engages (compare against native with
  `normal_precision='Precise'`);
- LSMC memory fields: training numbers unchanged, checkpoint forced;
- plan API: plan attributes, value/gradient parity at several points,
  curvature bumps, segmented settings (segment_steps/first_path).

Non-oracle unit tests pin dal-jax-only behaviour (filtering, errors).

## Documentation

CHANGELOG entry; README API table gains the new entry points.
