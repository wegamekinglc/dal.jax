# Portfolio Risk API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mirror `ScriptPortfolio_New` + `PortfolioMonteCarlo_ValueWithJacobianRisk/WithWeightedRisk` with native-exact semantics over the PR-#9 risk machinery.

**Architecture:** One new module `src/dal_jax/mc/portfolio.py` (portfolio dataclass, per-trade assembly, validation), thin api.py shims, oracle tests in `tests/oracle/test_dal_portfolio.py`. Binding contract = the dal-python 2026.11.1 wheel; probe before every behavior.

**Spec:** `docs/superpowers/specs/2026-11-09-portfolio-risk-design.md` (approved).

---

### Task 1 (only task): the portfolio risk API

**Step 0 — Probes (report findings; they override everything below):**
1. `model_owners` semantics: same model object passed twice → owners `[0, 0]` (identity dedup, already observed); equal-but-distinct objects → `[0, 1]`? Distinct types? Pin the rule.
2. Cross-trade jacobian structure: 2 trades × BS models — `jacobian` shape (rows=trades' outputs, cols=concatenated inputs?); are cross blocks exactly zero?
3. Input/output axis label collisions: two trades both exposing `spot` — how do the concatenated axes disambiguate (ids? prefixes? duplicate labels kept?)? Probe `complete_input_axis` ids and labels.
4. `PortfolioJacobianRiskRequest_` / `PortfolioWeightedRiskRequest_` field sets and defaults; how inputs/outputs selectors address cross-trade axes (global index? per-trade?).
5. Validation messages: empty lists, mismatched lengths (already pinned), duplicate trade_ids?, non-str trade_ids, models of different types (BS + CorrelatedBS mixed?).
6. Weighted semantics at portfolio level: weights vs trades vs outputs; aggregated value formula.
7. `PortfolioRiskExecution_` / `PortfolioRiskProvenance_` attr sets (dir()).
8. Shared valuation settings across trades: one `valuation` for all trades (assumed) — confirm; and EXERCISE/expired trades inside a portfolio: guarded or not?
9. Sobol sharing: do trades share the same random draws (same path ids) or get offset streams? (Values per trade vs individually-valued products will reveal.)

**Step 1 — TDD:** failing oracle tests covering (per probes): construction + validation parity; values/axes/jacobian parity on a 2-trade distinct-model portfolio AND a shared-model portfolio; selector filtering; weighted aggregation; execution/provenance attr-set equality (live dir() comparison, the PR-#9 drift-detector pattern); error messages both-sides.

**Step 2 — Implement** per spec: `ScriptPortfolioData` (trade_ids, products, models, model_owners, name), `ScriptPortfolio_New` validation, `portfolio_jacobian_result` (per-trade `jacobian_risk_result`-equivalents assembled; cross blocks per probe 2), weighted variant, dataclasses mirroring probe 7's attr sets exactly, api shims + aliases + `__all__`.

**Step 3 — Fold-ins from the polish review (same commit or a second one):**
- `src/dal_jax/mc/risk.py` `_described` (~line 252): builtin predicate for the value segment (`value is None or isinstance(value, (str, bool, int, float))`) so numpy-integer report_factors match native's omitted segment; pin with the probe case (`report_factors=[np.int64(2)]` element-type error message).
- Spec-note quote fix: "Tape capacity budget exceeded **before replay** [...]" in the 2026-11-08 spec's as-built note.
- Portfolio spec doc: record the identity-dedup observation from probe 1.

**Step 4 — Verify:** full targeted suites (`tests/oracle/test_dal_portfolio.py tests/oracle/test_dal_risk_api.py tests/mc tests/test_api_value.py`); gates ×4 (ruff format/check, lizard, pylint -E); backend-init still False.

**Step 5 — Commit(s)** with standard trailer; report probes + evidence.
