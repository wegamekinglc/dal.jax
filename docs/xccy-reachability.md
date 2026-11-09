# XCCY domain reachability — dal-python 2026.11.1

Date: 2026-11-09
Source: systematic probe of the pinned wheel (dal-python 2026.11.1) plus the
`dal-python-v2026.11.1` C++ source; grid-search over ~220 constructor
combinations for the blocking gate. Produced by the continuation program's
sub-project ③ investigation; the decision was to document rather than
implement (see below).

## Summary

The cross-currency domain's **valuation surface is unreachable from Python**,
so dal-jax does not mirror it. What IS reachable is (a) the input-spec layer
(21 constructors whose outputs feed only the blocked paths) and (b) a
complete market-calibration pipeline (`CalibrateXccyMarket`) — the latter is
calibration machinery that the handoff keeps out of scope. This document
records the evidence so nobody re-derives it, and the map for the day the
gate opens.

## The blocking gate

All three valuation entry points — `XccyInstrument_ValueByMonteCarlo`,
`XccyInstrument_ValueByGaussian`, `XccyBasket_PrepareMonteCarlo` — consume a
`CalibrationInstrument_` obtainable only from `XccyScriptInstrument_New` (or
the IR factories). The typed-handle wall that blocks the IR instrument
factories does NOT apply here: `CrossCurrencySwapConfig_`/`CrossCurrencySwap_`
construct fine from Python. Instead the factory rejects at a deeper,
**semantic** layer:

```
/import/rate; expected native projection dates/accrual/collateral exactly
matching canonical rate
```

Two sub-gates were cracked and are satisfiable: fixing identities must be
per-leg currency-matched canonical spellings (`IR:USD,LIBOR_3M_LCH`; LCH
whitelist only — the grammar is dal-jax's own `src/dal_jax/index.py`
TRADED_RATES table), and fixing hour/minute must be 0/0. The third is not:
the check compares the swap's embedded native rate-index name against a
canonical name regenerated from `Ccy::Conventions` stored facts
(`LiborIndex`/`LiborFixDays`/`LiborDayBasis`) that Python never exposes. A
grid search over every exposed knob (day basis × collateral {OIS, Libor3M,
Libor6M} × forecast tenor {3M, 6M} × fixing lag {0,1,2} × spot lag {0,2} ×
use-projection-curve × discount route × USD/EUR and USD/GBP pairs) left the
error invariant. Effectively `U` unless a future DAL release exposes the
convention facts or relaxes the check.

## Classification (65 names)

- **R — reachable (21):** `XccyScriptImportSettings_`(+`_New`),
  `XccyScriptContext_`(+`_New`), `XccyGaussianMarketSpec_`,
  `XccyGaussianDynamicsSpec_`(+`_New`), `XccyGaussianModelSpec_`,
  `XccyGaussianModel_New`, `XccyMonteCarloRequest_`,
  `XccyMonteCarloBasketRequest_`, `XccyMonteCarloReference_`,
  `XccyMonteCarloPrecision_`, `XccyInstrumentCalibrationSpec_`(+`_New`),
  `XccyTradeTerms_`, `XccyNotionalMode`, `XccyBasisCurveDeclaration_`,
  `JointXccyCalibrationSpecBuilder_`, `CalibrateXccyMarket` (works
  end-to-end). All outputs feed only blocked paths (or calibration).
- **T — telemetry/output types (28):** the 20-name `XccyMcBasket*` family,
  `XccyGaussianModelData_`, `XccyGaussianValue_`, the `XccyMonteCarlo*Result_`
  family, `PreparedXccyMonteCarloBasket_`, calibration result types.
- **U — unreachable (10):** `XccyScriptInstrument_New` (the gate),
  `XccyInstrument_ValueByGaussian`, `XccyInstrument_ValueByMonteCarlo`,
  `XccyBasket_PrepareMonteCarlo`, `XccyMonteCarloResult_Validate`,
  `Calibrate_XccyInstruments`, `Evaluate_XccyInstruments`,
  `CalibrationInstrument_Get_Xccy*Capabilities` (×2),
  `XccyBasketCalibrationError`.
- **C — calibration machinery (4):** `CalibrateJointXccyMarket`,
  `ValidateJointXccyAnalyticEligibility`, `BuildJointXccyQuoteRiskProvenance`,
  `BuildStagedXccyBasisQuoteRiskProvenance`.

## Minimal working examples (verified against the wheel)

Kept verbatim from the probe log so the map is reproducible; `DT` is a
`DateTime_` at midnight, `dates` a strictly-increasing date list, `dfs` log
discount factors:

```python
dc = dal.DiscountLogDF_New('DFS', 'USD', dates, dfs)          # per-currency curve
dc2 = dal.DiscountLogDF_New('DFS2', 'EUR', dates, dfs)
blk, blk2 = dal.CurveBlock_New(discount_curve=dc), dal.CurveBlock_New(discount_curve=dc2)
mkt = dal.CrossCurrencyMarket_New(domestic_block=blk, foreign_block=blk2,
                                  fx_spot=1.08, valuation_time=DT, collateral_currency='USD')
snap = dal.MarketFixingSnapshot_New({'IR:USD,LIBOR_3M_LCH': {DT: 0.043},
                                     'IR:EUR,LIBOR_3M_LCH': {DT: 0.039}})
cfg = dal.CrossCurrencySwapConfig_()                          # mutable convention tree
swap = dal.CrossCurrencySwap_New(dal.Date_(2024, 1, 2), dal.Date_(2024, 1, 4),
                                 dal.Date_(2029, 1, 4), 0.001, cfg)
mks = dal.XccyGaussianMarketSpec_(valuation_time=DT, pair=..., collateral_currency='USD',
                                  fx_spot=1.08, domestic_block=blk, foreign_block=blk2,
                                  basis=dc, discount_route='domestic')   # basis=collateral ccy
dyn = dal.XccyGaussianDynamicsSpec_(domestic_a=0.01, foreign_a=0.01,
                                    bucket_end_dates=[dal.Date_(2027, 1, 4)],
                                    domestic_sigma=[0.01], foreign_sigma=[0.01], fx_sigma=[0.01],
                                    factor_names=['rate:USD', 'rate:EUR', 'fx:EUR/USD'],
                                    correlation=[[1, .5, .3], [.5, 1, .2], [.3, .2, 1]])
model = dal.XccyGaussianModel_New('xg', dal.XccyGaussianModelSpec_(market=mks, dynamics=dyn))
sampling = dal.CalibrationMonteCarloSampling_(stream_version='2', paths_per_batch=128,
                                              batches=2, seed=42)       # v2 only
request = dal.XccyMonteCarloRequest_(measure_id='pv', phase='Fit', sampling=sampling)
builder = dal.CrossCurrencyCalibrationSpecBuilder_()          # then set fields; knot dates
spec = builder.Build()                                        # strictly after valuation
result = dal.CalibrateXccyMarket(spec)                        # works; .market/.fx_forward_curve
```

## Notes for the day the gate opens

- The Gaussian model spec layer is complete and constructible; a dal-jax
  mirror would need the 3-factor (domestic/foreign/FX) Gaussian dynamics,
  bucketed volatilities and **collateral-currency numeraire routing** — new
  engine concepts beyond the current Hybrid model (effort M).
- `CalibrationMonteCarloSampling_` is versioned; only `stream_version='2'`
  feeds the XCCY requests.
- `XccyGaussianModelSpec` factor names are currency-qualified
  (`rate:USD`, `fx:EUR/USD`); the error message's `rate:domestic` is a
  pattern, not a literal.
- `CrossCurrencyMarket_` and `CrossCurrencySwap_` are opaque (`dir()` empty)
  — only their factory inputs are inspectable.
