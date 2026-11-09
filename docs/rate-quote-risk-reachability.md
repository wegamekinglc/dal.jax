# Rate quote-risk provenance domain — reachability and deferral

Date: 2026-11-09
Source: systematic probe of the pinned dal-python 2026.11.1 wheel (every
chain executed from scratch; the tag's own Python test suites
`test_quote_risk.py` / `test_joint_quote_risk.py` / `test_rate_curvature.py`
were independently re-derived). Sub-project ⑤ of the continuation program;
decision: document and defer (rationale below).

## Summary

Unlike the IR-instrument and XCCY clusters, **nothing here is gated** — the
whole domain is Python-reachable and fully exercisable. It is deferred
because it is 100% downstream of **curve calibration plus a rate-trade
pricing engine (deposit/IRS pricers with AAD)**, neither of which dal-jax
has; the project handoff keeps calibration out of scope. Numerical parity
would require building that substrate first (sized L+ by itself); only
API-shape dataclasses are mirrorable without it, which would be facade code.

## What works natively (minimal chain, verbatim)

```python
today = dal.Date_(2025, 6, 20); spot = today.AddDays(2)
maturities = [spot.AddDays(y * 365) for y in (2, 5, 10)]
fixed_leg = dal.RateLegConvention_New(dal.PeriodLength_New("6M"), dal.DayBasis_New("ACT_365F"))
float_leg = dal.RateLegConvention_New(dal.PeriodLength_New("3M"), dal.DayBasis_New("ACT_360"))
ois_idx = dal.RateIndexConvention_New(dal.PeriodLength_New("12M"), dal.DayBasis_New("ACT_360"), dal.CollateralType_OIS())
b = dal.CurveCalibrationSpecBuilder_()
b.today_ = today; b.ccy_ = dal.String_("USD"); b.curveName_ = dal.String_("c1")
b.calibrateDiscountCurve_ = True; b.initialGuess_ = 0.04
b.instruments_ = [dal.OISSwap_New(today, spot, m, 0.04, fixed_leg, ois_idx, float_leg)
                  for m in maturities]
b.knotDates_ = maturities
spec = b.Build()
options = dal.CurveCalibrationOptions_(); options.compute_eff_jacobian_inverse = True
calibrated = dal.CalibrateSingleCurve(spec, options)
market = dal.RatePricingMarket_(valuation_time=dal.DateTime_(today, 9, 0), result_currency="USD",
                                curve_components={"discount": calibrated.curve_},
                                fixings=dal.MarketFixingSnapshot_New({}))
config = dal.RateQuoteRiskProvenanceConfig_(calibration_id="probe",
                                            component_key_by_parameter_block={"c1": "discount"})
prov = dal.BuildSingleCurveQuoteRiskProvenance(spec=spec, result=calibrated, options=options,
                                               bound_market=market, config=config)
terms = dal.DepositTradeTerms_(notional=1e6, contract_rate=0.022, lend=True, index=index,
                               discount_component_key="discount")
trade = dal.RateTradeDefinition_(instrument_id="d1", instrument_type=dal.RateInstrumentType.DEPOSIT,
                                 trade_date=today, start_date=today, maturity_date=maturities[0],
                                 currency="USD", terms=terms)
agg = dal.AggregateRatePortfolioQuoteRisk(trades=[trade], market=market, provenances=[prov])
node = dal.AggregateRatePortfolioNodeRisk(trades=[trade], market=market, component_keys=["discount"])
```

All four provenance builders were exercised (the domain turned out to have
four, not two): SINGLE_CURVE (above), STAGED_XCCY_BASIS (fed directly by the
XCCY chain from `docs/xccy-reachability.md` — `CalibrateXccyMarket`'s
`.basis_curve`/`.market` slot into `RatePricingMarket_`), JOINT_XCCY, and
JOINT_MULTI_CURVE.

## Classification (26 names)

- **R (10):** `RateQuoteRiskProvenanceConfig_` (2 kw-only overloads),
  `RateTradeQuoteCurvatureSettings_`, the four `Build*QuoteRiskProvenance`
  functions, `AggregateRatePortfolioNodeRisk` / `AggregateRatePortfolioQuoteRisk`,
  the two scheme functions (`RateQuoteRiskAxisFingerprintScheme` /
  `RateQuoteRiskStateFingerprintScheme` — plain functions returning URN
  strings like `dal.quote-risk-axis/1+jcs+sha256`, not enums).
- **T (16):** the output-only provenance/state/axis/coordinate/bucket types,
  `RatePortfolioNodeRisk_`(+Component/MetaEntry), `RatePortfolioQuoteRisk_`,
  `RateTradeQuoteCurvatureResult_`. Fully inspectable, no constructors.
- **U (0), opaque (0).** The adjacent `PortfolioRiskProvenance_` /
  `RiskResultProvenance_` are the MC side, already mirrored by dal-jax.

## Notable semantics (for the day this is implemented)

- Quote pullback: `dPV/dquote = dPV/dnode @ effective_inverse`; buckets carry
  `dv01 == d_pv_d_decimal_quote * 1e-4` exactly; aggregation policy
  `UnconvertedByActualPvCcy` with structural zeros and failure reporting
  (`provenance_failures`, reason `QUOTE_RISK_INVERSE_NOT_REQUESTED` when the
  option is off).
- The `retain_calibration_record=True` config gates the
  `CalibrationPullback_New` bridge (else
  `QUOTE_RISK_CALIBRATION_RECORD_NOT_RETAINED`); `calibration_record` is a
  JSON string (~2.8 KB), and `effective_inverse` returns mutation-safe
  copies.
- `RateCalibration_New(spec)` is a passive snapshot factory that builds
  SINGLE_CURVE provenance internally — the only input
  `RateTradeQuoteCurvature` accepts (no market assembly needed);
  `curvature settings.weights` must match the trade count.
- All five factory/aggregation functions are kw-only by design.

## Reopen conditions

Implement when (and only when) curve calibration enters dal-jax's scope —
the substrate this domain is downstream of. Probe scripts preserved at
`/tmp/probe_qr1..9.py` at the time of writing; C++ references:
`dal-python/src/bindings/curve.cpp:2048-2240`,
`dal-cpp/dal/curve/quoteriskprovenance.cpp:1832-1920`.
