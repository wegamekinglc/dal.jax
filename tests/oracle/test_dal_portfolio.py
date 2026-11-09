"""Portfolio risk API parity with dal-python 2026.11.1."""

import dataclasses
import re

import numpy as np
import pytest

from dal_jax.api import (
    BSModelData_New,
    CorrelatedBSModelData_New,
    JacobianRiskRequest_,
    MonteCarlo_ValueWithJacobianRisk,
    PortfolioJacobianRiskRequest_,
    PortfolioMonteCarlo_ValueWithJacobianRisk,
    PortfolioMonteCarlo_ValueWithWeightedRisk,
    PortfolioWeightedRiskRequest_,
    Product_New,
    RiskRequest_,
    ScriptPortfolio_New,
    ValuationSettings,
)
from dal_jax.dates import Date
from dal_jax.errors import (
    InvalidPathCount,
    InvalidPortfolioJacobianRequest,
    InvalidPortfolioRiskRequest,
    InvalidPortfolioWeightedRequest,
    InvalidRiskRequest,
    InvalidScriptPortfolio,
    InvalidSetting,
    UnsupportedPortfolioRisk,
)
from dal_jax.script.product import ScriptProductSettings

pytestmark = pytest.mark.oracle

_MODEL = (100.0, 0.2, 0.03, 0.01)
_MODEL2 = (90.0, 0.3, 0.03, 0.01)
_VALUATION = ValuationSettings(evaluation_date=Date.ymd(2022, 9, 15))

#  Probe-verified 64-path reference values for the fixtures below (sobol, BS).
SIMPLE = ([(2024, 1, 2)], ["p pays spot()"])
SIMPLE_VALUE = 97.84308474694558
TWO_SLOT = ([(2024, 1, 2)], ["a = 1\nb = 2\na pays spot()\nb pays spot() * 2"])
TWO_SLOT_PAYOFF_VALUE = 177.07847230924992  # slot b: its initial 2 plus spot() * 2
TWO_SLOT_A_VALUE = 88.53923615462496  # slot a: its initial 1 plus spot()
CONSTANT = (["K", (2024, 1, 2)], ["100", "p pays MAX(SPOT()-K,0)"])


def _cells(dates):
    return [Date.ymd(*d) if isinstance(d, tuple) else d for d in dates]


def _ours_products(specs):
    return [Product_New(_cells(dates), list(events)) for dates, events in specs]


def _native_products(dal, specs):
    return [
        dal.Product_New(
            [dal.Date_(*d) if isinstance(d, tuple) else d for d in dates],
            list(events),
            settings=dal.ScriptProductSettings_(),
        )
        for dates, events in specs
    ]


def _native_valuation(dal):
    return dal.ScriptValuationSettings_(evaluation_date=dal.Date_(2022, 9, 15))


def _ours_portfolio(specs, models):
    ids = [f"t{position + 1}" for position in range(len(specs))]
    return ScriptPortfolio_New(ids, _ours_products(specs), list(models))


def _native_portfolio(dal, specs, models):
    ids = [f"t{position + 1}" for position in range(len(specs))]
    return dal.ScriptPortfolio_New(ids, _native_products(dal, specs), list(models))


def _ours_models(*tuples):
    return [BSModelData_New(*values) for values in tuples]


def _native_models(dal, *tuples):
    return [dal.BSModelData_New(*values) for values in tuples]


def _input_axis_records(result):
    return [
        (c.family, c.id, c.label, c.ordinal, c.native_unit, c.physical_unit, c.report_scale)
        for c in result.input_axis
    ]


def _output_records(coordinates):
    return [(c.id, c.label, c.slot) for c in coordinates]


def _assert_group_parity(ours, theirs):
    assert len(ours.execution.groups) == len(theirs.execution.groups)
    for ours_group, theirs_group in zip(ours.execution.groups, theirs.execution.groups):
        assert ours_group.model_owner == theirs_group.model_owner
        assert ours_group.trade_positions == list(theirs_group.trade_positions)
        assert ours_group.replay_attempts == theirs_group.replay_attempts
        assert list(ours_group.actual_widths) == list(theirs_group.actual_widths)
        assert ours_group.evaluator_calls == theirs_group.evaluator_calls
        assert ours_group.generated_scenarios == theirs_group.generated_scenarios
        assert ours_group.prefix_reversals == theirs_group.prefix_reversals
        assert ours_group.suffix_reversals == theirs_group.suffix_reversals
        assert ours_group.factors == theirs_group.factors
        assert ours_group.random_dimension == theirs_group.random_dimension
        assert ours_group.deterministic_numeraire == theirs_group.deterministic_numeraire
        assert list(ours_group.timeline) == list(theirs_group.timeline)


def _assert_portfolio_provenance(ours, theirs, method):
    assert ours.provenance.method == theirs.provenance.method == method
    assert ours.provenance.normalization == theirs.provenance.normalization
    assert ours.provenance.calibration == theirs.provenance.calibration
    assert ours.provenance.trade_ids == list(theirs.provenance.trade_ids)
    assert ours.provenance.model_owners == list(theirs.provenance.model_owners)
    assert str(ours.provenance.evaluation_date) == str(theirs.provenance.evaluation_date)
    assert [t.model_type for t in ours.provenance.trades] == [
        t.model_type for t in theirs.provenance.trades
    ]
    assert [t.method for t in ours.provenance.trades] == [
        t.method for t in theirs.provenance.trades
    ]


def _assert_jacobian_parity(ours, theirs, *, rtol=1e-9):
    np.testing.assert_allclose(ours.values, theirs.values, rtol=rtol)
    np.testing.assert_allclose(
        np.asarray(ours.jacobian), np.asarray(theirs.jacobian.to_rows()), rtol=rtol, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(ours.reported_jacobian),
        np.asarray(theirs.reported_jacobian.to_rows()),
        rtol=rtol,
        atol=1e-12,
    )
    assert _input_axis_records(ours) == _input_axis_records(theirs)
    assert [c.id for c in ours.complete_input_axis] == [c.id for c in theirs.complete_input_axis]
    assert _output_records(ours.output_axis) == _output_records(theirs.output_axis)
    assert _output_records(ours.complete_output_axis) == _output_records(
        theirs.complete_output_axis
    )
    _assert_group_parity(ours, theirs)
    _assert_portfolio_provenance(ours, theirs, "NativeAADJacobianPortfolio")
    assert ours.execution.requested_max_block_width == theirs.execution.requested_max_block_width


def _assert_weighted_parity(ours, theirs, *, rtol=1e-9):
    assert ours.weighted_value == pytest.approx(theirs.weighted_value, rel=rtol)
    np.testing.assert_allclose(ours.component_means, theirs.component_means, rtol=rtol)
    np.testing.assert_allclose(ours.weights, theirs.weights, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(ours.jacobian), np.asarray(theirs.jacobian.to_rows()), rtol=rtol, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(ours.reported_jacobian),
        np.asarray(theirs.reported_jacobian.to_rows()),
        rtol=rtol,
        atol=1e-12,
    )
    assert _input_axis_records(ours) == _input_axis_records(theirs)
    assert [c.id for c in ours.complete_input_axis] == [c.id for c in theirs.complete_input_axis]
    assert _output_records(ours.output_axis) == _output_records(theirs.output_axis)
    assert _output_records(ours.complete_output_axis) == _output_records(
        theirs.complete_output_axis
    )
    _assert_group_parity(ours, theirs)
    _assert_portfolio_provenance(ours, theirs, "NativeAADWeightedPortfolio")
    assert ours.execution.requested_max_block_width == theirs.execution.requested_max_block_width


def _ours_jacobian(portfolio, n_paths=64, *, request=None):
    return PortfolioMonteCarlo_ValueWithJacobianRisk(
        portfolio, n_paths, valuation=_VALUATION, request=request
    )


def _theirs_jacobian(dal, portfolio, n_paths=64, *, request=None):
    return dal.PortfolioMonteCarlo_ValueWithJacobianRisk(
        portfolio, n_paths, valuation=_native_valuation(dal), request=request
    )


def _ours_weighted(portfolio, n_paths=64, *, request=None):
    return PortfolioMonteCarlo_ValueWithWeightedRisk(
        portfolio, n_paths, valuation=_VALUATION, request=request
    )


def _theirs_weighted(dal, portfolio, n_paths=64, *, request=None):
    return dal.PortfolioMonteCarlo_ValueWithWeightedRisk(
        portfolio, n_paths, valuation=_native_valuation(dal), request=request
    )


# --------------------------------------------------------------------------- #
# ScriptPortfolio_New
# --------------------------------------------------------------------------- #


def test_script_portfolio_new_exposes_native_fields(dal):
    ours = _ours_portfolio([SIMPLE, SIMPLE], _ours_models(_MODEL, _MODEL2))
    theirs = _native_portfolio(dal, [SIMPLE, SIMPLE], _native_models(dal, _MODEL, _MODEL2))
    assert ours.trade_ids == list(theirs.trade_ids) == ["t1", "t2"]
    assert ours.model_owners == list(theirs.model_owners) == [0, 1]
    assert ours.name == theirs.name == ""
    assert ours.type == theirs.type == "ScriptPortfolioData"


def test_script_portfolio_owners_dedup_by_identity(dal):
    #  Identity dedup: the same object shares an owner, equal-but-distinct
    #  objects do not, and a model reused after a gap keeps its first owner.
    same = BSModelData_New(*_MODEL)
    equal = BSModelData_New(*_MODEL)
    other = BSModelData_New(*_MODEL2)
    products = _ours_products([SIMPLE, SIMPLE, SIMPLE])
    ids = ["t1", "t2", "t3"]
    for models, owners in (
        ([same, same], [0, 0]),
        ([same, equal], [0, 1]),
        ([same, other, same], [0, 1, 0]),
        ([same, other, other], [0, 1, 1]),
    ):
        ours = ScriptPortfolio_New(ids[: len(models)], products[: len(models)], models)
        assert ours.model_owners == owners
    native_same = dal.BSModelData_New(*_MODEL)
    native_equal = dal.BSModelData_New(*_MODEL)
    theirs = dal.ScriptPortfolio_New(
        ["t1", "t2"],
        _native_products(dal, [SIMPLE, SIMPLE]),
        [native_same, native_same],
    )
    assert list(theirs.model_owners) == [0, 0]
    theirs_distinct = dal.ScriptPortfolio_New(
        ["t1", "t2"],
        _native_products(dal, [SIMPLE, SIMPLE]),
        [native_same, native_equal],
    )
    assert list(theirs_distinct.model_owners) == [0, 1]


CTOR_ERRORS = [
    pytest.param(
        "model-single",
        lambda p, m: (["t1"], [p], m),
        InvalidScriptPortfolio,
        "ScriptPortfolio_New; modelData; type=ModelData_; expected list or tuple, or None",
        id="model-single",
    ),
    pytest.param(
        "model-none",
        lambda p, m: (["t1"], [p], None),
        InvalidScriptPortfolio,
        "ScriptPortfolio_New; modelData; type=NoneType, value=None; expected list or tuple, or None",
        id="model-none",
    ),
    pytest.param(
        "length-mismatch",
        lambda p, m: (["t1", "t2"], [p, p], [m]),
        InvalidScriptPortfolio,
        "trade_ids, products and modelData must have equal nonzero lengths",
        id="length-mismatch",
    ),
    pytest.param(
        "empty-lists",
        lambda p, m: ([], [], []),
        InvalidScriptPortfolio,
        "trade_ids, products and modelData must have equal nonzero lengths",
        id="empty-lists",
    ),
    pytest.param(
        "trade-ids-none",
        lambda p, m: (None, [p], [m]),
        InvalidScriptPortfolio,
        "trade_ids, products and modelData must have equal nonzero lengths",
        id="trade-ids-none",
    ),
    pytest.param(
        "dup-ids",
        lambda p, m: (["t1", "T1"], [p, p], [m, m]),
        InvalidScriptPortfolio,
        "trade ID must be nonempty and unique; field=tradeIds; trade=T1",
        id="dup-ids",
    ),
    pytest.param(
        "empty-id",
        lambda p, m: ([""], [p], [m]),
        InvalidScriptPortfolio,
        "trade ID must be nonempty and unique; field=tradeIds; trade=",
        id="empty-id",
    ),
    pytest.param(
        "non-str-id",
        lambda p, m: (["t1", 2], [p, p], [m, m]),
        InvalidScriptPortfolio,
        "ScriptPortfolio_New; trade_ids; type=int, value=2; expected str or DAL String_ without NUL",
        id="non-str-id",
    ),
    pytest.param(
        "nul-id",
        lambda p, m: (["t1\0"], [p], [m]),
        InvalidScriptPortfolio,
        "ScriptPortfolio_New; trade_ids; type=str, value='t1\\x00'; expected str or DAL String_ without NUL",
        id="nul-id",
    ),
    pytest.param(
        "trade-ids-string",
        lambda p, m: ("t1", [p], [m]),
        InvalidScriptPortfolio,
        "ScriptPortfolio_New; trade_ids; type=str, value='t1'; expected list or tuple, or None",
        id="trade-ids-string",
    ),
    pytest.param(
        "products-none",
        lambda p, m: (["t1"], None, [m]),
        InvalidScriptPortfolio,
        "ScriptPortfolio_New; products; type=NoneType, value=None; expected list or tuple, or None",
        id="products-none",
    ),
    pytest.param(
        "products-single",
        lambda p, m: (["t1"], p, [m]),
        InvalidScriptPortfolio,
        "ScriptPortfolio_New; products; type=ScriptProductData_; expected list or tuple, or None",
        id="products-single",
    ),
    pytest.param(
        "product-element",
        lambda p, m: (["t1"], [42], [m]),
        InvalidScriptPortfolio,
        "ScriptPortfolio_New; trade=t1; products; type=int, value=42; expected ScriptProductData_",
        id="product-element",
    ),
    pytest.param(
        "model-element",
        lambda p, m: (["t1"], [p], [42]),
        InvalidScriptPortfolio,
        "ScriptPortfolio_New; trade=t1; modelData; type=int, value=42; expected ModelData_",
        id="model-element",
    ),
]


@pytest.mark.parametrize(("label", "args_of", "error_class", "detail"), CTOR_ERRORS)
def test_script_portfolio_new_rejects_bad_inputs(dal, label, args_of, error_class, detail):
    message = f"InvalidScriptPortfolio: {detail}"
    native_product = _native_products(dal, [SIMPLE])[0]
    native_model = _native_models(dal, _MODEL)[0]
    ids, products, models = args_of(native_product, native_model)
    with pytest.raises((RuntimeError, TypeError), match=re.escape(message)):
        dal.ScriptPortfolio_New(ids, products, models)
    our_product = _ours_products([SIMPLE])[0]
    our_model = _ours_models(_MODEL)[0]
    ids, products, models = args_of(our_product, our_model)
    with pytest.raises(error_class, match=re.escape(message)) as error:
        ScriptPortfolio_New(ids, products, models)
    assert str(error.value) == message


def test_script_portfolio_new_accepts_tuples(dal):
    products = _ours_products([SIMPLE, SIMPLE])
    models = _ours_models(_MODEL, _MODEL2)
    ours = ScriptPortfolio_New(("t1", "t2"), tuple(products), tuple(models))
    assert ours.model_owners == [0, 1]
    theirs = dal.ScriptPortfolio_New(
        ("t1", "t2"),
        tuple(_native_products(dal, [SIMPLE, SIMPLE])),
        tuple(_native_models(dal, _MODEL, _MODEL2)),
    )
    assert list(theirs.model_owners) == [0, 1]


# --------------------------------------------------------------------------- #
# Jacobian variant
# --------------------------------------------------------------------------- #


def test_portfolio_jacobian_matches_native(dal):
    ours = _ours_jacobian(_ours_portfolio([SIMPLE, TWO_SLOT], _ours_models(_MODEL, _MODEL2)))
    theirs = _theirs_jacobian(
        dal, _native_portfolio(dal, [SIMPLE, TWO_SLOT], _native_models(dal, _MODEL, _MODEL2))
    )
    _assert_jacobian_parity(ours, theirs)
    #  Probe-embedded anchors: slot values include each variable's initial
    #  assignment, and the 2x8 jacobian is block-diagonal across owners.
    assert ours.values[0] == pytest.approx(SIMPLE_VALUE, rel=1e-9)
    assert ours.values[1] == pytest.approx(TWO_SLOT_PAYOFF_VALUE, rel=1e-9)
    rows = np.asarray(ours.jacobian)
    assert rows.shape == (2, 8)
    np.testing.assert_allclose(rows[0, 4:], 0.0, atol=1e-12)
    np.testing.assert_allclose(rows[1, :4], 0.0, atol=1e-12)
    assert [c.id for c in ours.complete_output_axis] == [
        "trade:0:payoff",
        "trade:1:output:0",
        "trade:1:payoff",
    ]
    assert _output_records(ours.output_axis) == [
        ("trade:0:payoff", "p", 0),
        ("trade:1:payoff", "b", 1),
    ]


def test_portfolio_jacobian_shared_model_merges_owner(dal):
    doubled = ([(2024, 1, 2)], ["q pays spot() * 2"])
    model = BSModelData_New(*_MODEL)
    native_model = dal.BSModelData_New(*_MODEL)
    ours = _ours_jacobian(_ours_portfolio([SIMPLE, doubled], [model, model]))
    theirs = _theirs_jacobian(
        dal, _native_portfolio(dal, [SIMPLE, doubled], [native_model, native_model])
    )
    _assert_jacobian_parity(ours, theirs)
    assert ours.provenance.model_owners == [0, 0]
    assert [c.id for c in ours.complete_input_axis] == [f"model:0:parameter:{i}" for i in range(4)]
    #  One merged group for both trades, and both rows live in the shared
    #  owner's columns; the second trade pays exactly twice the first.
    (group,) = ours.execution.groups
    assert group.trade_positions == [0, 1]
    assert group.replay_attempts == 2
    rows = np.asarray(ours.jacobian)
    assert rows.shape == (2, 4)
    np.testing.assert_allclose(rows[1], 2.0 * rows[0], rtol=1e-12)
    np.testing.assert_allclose(ours.values[1], 2.0 * ours.values[0], rtol=1e-12)


def test_portfolio_jacobian_same_model_different_dates_split_groups(dal):
    #  Groups merge only when owner AND timeline coincide.
    later = ([(2025, 6, 2)], ["r pays spot()"])
    model = BSModelData_New(*_MODEL)
    native_model = dal.BSModelData_New(*_MODEL)
    ours = _ours_jacobian(_ours_portfolio([SIMPLE, later], [model, model]))
    theirs = _theirs_jacobian(
        dal, _native_portfolio(dal, [SIMPLE, later], [native_model, native_model])
    )
    _assert_jacobian_parity(ours, theirs)
    assert [group.trade_positions for group in ours.execution.groups] == [[0], [1]]


def test_portfolio_jacobian_axis_includes_trade_constants(dal):
    ours = _ours_jacobian(_ours_portfolio([CONSTANT], _ours_models(_MODEL)))
    theirs = _theirs_jacobian(dal, _native_portfolio(dal, [CONSTANT], _native_models(dal, _MODEL)))
    _assert_jacobian_parity(ours, theirs)
    assert [c.id for c in ours.complete_input_axis] == [
        "model:0:parameter:0",
        "model:0:parameter:1",
        "model:0:parameter:2",
        "model:0:parameter:3",
        "trade:0:constant:0",
    ]
    constant = ours.complete_input_axis[-1]
    assert (constant.family, constant.label, constant.native_unit, constant.value) == (
        "constant",
        "K",
        "script-number",
        100.0,
    )


def test_portfolio_jacobian_values_equal_per_trade_valuations(dal):
    #  Each group restarts the sobol stream, so a trade's portfolio value is
    #  its standalone per-trade jacobian value.
    portfolio = _ours_portfolio([SIMPLE, TWO_SLOT], _ours_models(_MODEL, _MODEL2))
    ours = _ours_jacobian(portfolio)
    for values, (dates, events), model in zip(
        ([ours.values[0]], [ours.values[1]]),
        [SIMPLE, TWO_SLOT],
        _ours_models(_MODEL, _MODEL2),
    ):
        single = MonteCarlo_ValueWithJacobianRisk(
            Product_New(_cells(dates), list(events)), model, 64, valuation=_VALUATION
        )
        np.testing.assert_allclose(values, single.values, rtol=1e-12)


def test_portfolio_jacobian_result_attr_sets_match_native(dal):
    #  The PR-#9 drift-detector pattern: live dir() comparison per object.
    ours = _ours_jacobian(_ours_portfolio([SIMPLE, TWO_SLOT], _ours_models(_MODEL, _MODEL2)))
    theirs = _theirs_jacobian(
        dal, _native_portfolio(dal, [SIMPLE, TWO_SLOT], _native_models(dal, _MODEL, _MODEL2))
    )
    for ours_object, theirs_object in (
        (ours, theirs),
        (ours.execution, theirs.execution),
        (ours.provenance, theirs.provenance),
        (ours.execution.groups[0], theirs.execution.groups[0]),
        (ours.provenance.trades[0], theirs.provenance.trades[0]),
        (
            ours.execution.groups[0].sample_definitions[0],
            theirs.execution.groups[0].sample_definitions[0],
        ),
    ):
        assert sorted(f.name for f in dataclasses.fields(ours_object)) == sorted(
            attr for attr in dir(theirs_object) if not attr.startswith("_")
        )


def test_portfolio_jacobian_selects_inputs_in_request_order(dal):
    request = PortfolioJacobianRiskRequest_(inputs=["model:1:parameter:0", "model:0:parameter:1"])
    ours = _ours_jacobian(
        _ours_portfolio([SIMPLE, SIMPLE], _ours_models(_MODEL, _MODEL2)), request=request
    )
    theirs = _theirs_jacobian(
        dal,
        _native_portfolio(dal, [SIMPLE, SIMPLE], _native_models(dal, _MODEL, _MODEL2)),
        request=dal.PortfolioJacobianRiskRequest_(
            inputs=["model:1:parameter:0", "model:0:parameter:1"]
        ),
    )
    _assert_jacobian_parity(ours, theirs)
    assert [c.id for c in ours.input_axis] == ["model:1:parameter:0", "model:0:parameter:1"]
    rows = np.asarray(ours.jacobian)
    np.testing.assert_allclose(rows[0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(rows[1, 1], 0.0, atol=1e-12)


def test_portfolio_jacobian_selects_trade_outputs(dal):
    portfolio = _ours_portfolio([SIMPLE, TWO_SLOT], _ours_models(_MODEL, _MODEL2))
    native_portfolio = _native_portfolio(
        dal, [SIMPLE, TWO_SLOT], _native_models(dal, _MODEL, _MODEL2)
    )
    for outputs in (
        ["trade:0:payoff"],
        ["trade:1:output:0"],
        ["trade:0:payoff", "trade:1:output:0", "trade:1:payoff"],
        ["trade:1:payoff", "trade:0:payoff"],
    ):
        ours = _ours_jacobian(portfolio, request=PortfolioJacobianRiskRequest_(outputs=outputs))
        theirs = _theirs_jacobian(
            dal, native_portfolio, request=dal.PortfolioJacobianRiskRequest_(outputs=outputs)
        )
        _assert_jacobian_parity(ours, theirs)
        assert [c.id for c in ours.output_axis] == outputs
    inner = _ours_jacobian(
        portfolio, request=PortfolioJacobianRiskRequest_(outputs=["trade:1:output:0"])
    )
    assert inner.values == [pytest.approx(TWO_SLOT_A_VALUE, rel=1e-9)]
    all_outputs = _ours_jacobian(
        portfolio,
        request=PortfolioJacobianRiskRequest_(
            outputs=["trade:0:payoff", "trade:1:output:0", "trade:1:payoff"]
        ),
    )
    assert [
        (g.replay_attempts, g.evaluator_calls, g.generated_scenarios)
        for g in all_outputs.execution.groups
    ] == [
        (1, 64, 64),
        (2, 128, 128),
    ]
    #  A trade left unselected still owns a zeroed group.
    first_only = _ours_jacobian(
        portfolio, request=PortfolioJacobianRiskRequest_(outputs=["trade:0:payoff"])
    )
    stats = [
        (g.trade_positions, g.replay_attempts, g.evaluator_calls, g.generated_scenarios)
        for g in first_only.execution.groups
    ]
    assert stats == [([0], 1, 64, 64), ([1], 0, 0, 0)]


def test_portfolio_jacobian_mixed_model_types(dal):
    model = CorrelatedBSModelData_New(
        ["EQ[A]", "EQ[B]"], [100.0, 90.0], [0.2, 0.3], [0.01, 0.02], 0.03, [[1.0, 0.5], [0.5, 1.0]]
    )
    native_model = dal.CorrelatedBSModelData_New(
        ["EQ[A]", "EQ[B]"],
        [100.0, 90.0],
        [0.2, 0.3],
        [0.01, 0.02],
        0.03,
        dal.DoubleMatrix_([[1.0, 0.5], [0.5, 1.0]]),
    )
    indexed = ([(2024, 1, 2), (2024, 7, 2)], ["p pays spot()", "p pays spot() * 2"])
    ours = _ours_jacobian(
        ScriptPortfolio_New(
            ["t1", "t2"],
            [
                Product_New(_cells(SIMPLE[0]), list(SIMPLE[1])),
                Product_New(
                    _cells(indexed[0]),
                    list(indexed[1]),
                    settings=ScriptProductSettings(default_index="EQ[A]"),
                ),
            ],
            [BSModelData_New(*_MODEL), model],
        )
    )
    theirs = _theirs_jacobian(
        dal,
        dal.ScriptPortfolio_New(
            ["t1", "t2"],
            [
                _native_products(dal, [SIMPLE])[0],
                dal.Product_New(
                    [dal.Date_(*d) for d in indexed[0]],
                    list(indexed[1]),
                    settings=dal.ScriptProductSettings_(default_index="EQ[A]"),
                ),
            ],
            [dal.BSModelData_New(*_MODEL), native_model],
        ),
    )
    _assert_jacobian_parity(ours, theirs)
    assert [t.model_type for t in ours.provenance.trades] == [
        "BSModelData_",
        "CorrelatedBSModelData_",
    ]
    assert [c.id for c in ours.complete_input_axis] == [
        *(f"model:0:parameter:{i}" for i in range(4)),
        *(f"model:1:parameter:{i}" for i in range(7)),
    ]
    assert ours.execution.groups[1].factors == 2
    assert ours.execution.groups[1].random_dimension == 4


def test_portfolio_jacobian_scales_reported_jacobian_by_factors(dal):
    portfolio = _ours_portfolio([SIMPLE, SIMPLE], _ours_models(_MODEL, _MODEL2))
    native_portfolio = _native_portfolio(
        dal, [SIMPLE, SIMPLE], _native_models(dal, _MODEL, _MODEL2)
    )
    factors = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0]
    ours = _ours_jacobian(portfolio, request=PortfolioJacobianRiskRequest_(report_factors=factors))
    theirs = _theirs_jacobian(
        dal, native_portfolio, request=dal.PortfolioJacobianRiskRequest_(report_factors=factors)
    )
    _assert_jacobian_parity(ours, theirs)
    np.testing.assert_allclose(
        np.asarray(ours.reported_jacobian),
        np.asarray(ours.jacobian) * np.asarray(factors),
        rtol=1e-12,
    )


# --------------------------------------------------------------------------- #
# Weighted variant
# --------------------------------------------------------------------------- #


def test_portfolio_weighted_matches_native(dal):
    ours = _ours_weighted(_ours_portfolio([SIMPLE, TWO_SLOT], _ours_models(_MODEL, _MODEL2)))
    theirs = _theirs_weighted(
        dal, _native_portfolio(dal, [SIMPLE, TWO_SLOT], _native_models(dal, _MODEL, _MODEL2))
    )
    _assert_weighted_parity(ours, theirs)
    assert ours.weights == [1.0, 1.0]
    assert ours.weighted_value == pytest.approx(SIMPLE_VALUE + TWO_SLOT_PAYOFF_VALUE, rel=1e-9)
    #  Weighted groups never replay: one valuation per trade instead.
    for group in ours.execution.groups:
        assert group.replay_attempts == 0
        assert group.actual_widths == []
        assert group.evaluator_calls == 64
        assert group.generated_scenarios == 64
    assert ours.execution.requested_max_block_width == 0


def test_portfolio_weighted_result_attr_sets_match_native(dal):
    ours = _ours_weighted(_ours_portfolio([SIMPLE, TWO_SLOT], _ours_models(_MODEL, _MODEL2)))
    theirs = _theirs_weighted(
        dal, _native_portfolio(dal, [SIMPLE, TWO_SLOT], _native_models(dal, _MODEL, _MODEL2))
    )
    assert sorted(f.name for f in dataclasses.fields(ours)) == sorted(
        attr for attr in dir(theirs) if not attr.startswith("_")
    )


def test_portfolio_weighted_scales_with_weights(dal):
    portfolio = _ours_portfolio([SIMPLE, TWO_SLOT], _ours_models(_MODEL, _MODEL2))
    native_portfolio = _native_portfolio(
        dal, [SIMPLE, TWO_SLOT], _native_models(dal, _MODEL, _MODEL2)
    )
    base = _ours_weighted(portfolio)
    for weights in ([2.0, 3.0], [2, 3], (2.0, 3.0)):
        ours = _ours_weighted(portfolio, request=PortfolioWeightedRiskRequest_(weights=weights))
        theirs = _theirs_weighted(
            dal, native_portfolio, request=dal.PortfolioWeightedRiskRequest_(weights=weights)
        )
        _assert_weighted_parity(ours, theirs)
        assert ours.weights == [2.0, 3.0]
        assert ours.weighted_value == pytest.approx(
            2.0 * base.component_means[0] + 3.0 * base.component_means[1], rel=1e-12
        )
    for request in (PortfolioWeightedRiskRequest_(weights=None), PortfolioWeightedRiskRequest_()):
        ours = _ours_weighted(portfolio, request=request)
        _assert_weighted_parity(ours, _theirs_weighted(dal, native_portfolio))
        assert ours.weights == [1.0, 1.0]


def test_portfolio_weighted_selects_outputs_and_inputs(dal):
    portfolio = _ours_portfolio([SIMPLE, TWO_SLOT], _ours_models(_MODEL, _MODEL2))
    native_portfolio = _native_portfolio(
        dal, [SIMPLE, TWO_SLOT], _native_models(dal, _MODEL, _MODEL2)
    )
    cases = [
        {"outputs": ["trade:1:output:0"], "weights": [5.0]},
        {"outputs": ["trade:1:payoff", "trade:0:payoff"], "weights": [1.0, 2.0]},
        {"inputs": ["model:0:parameter:0"]},
        {"inputs": ["model:1:parameter:0", "model:0:parameter:1"], "weights": [2.0, 3.0]},
    ]
    for kwargs in cases:
        ours = _ours_weighted(portfolio, request=PortfolioWeightedRiskRequest_(**kwargs))
        theirs = _theirs_weighted(
            dal, native_portfolio, request=dal.PortfolioWeightedRiskRequest_(**kwargs)
        )
        _assert_weighted_parity(ours, theirs)
        if "outputs" in kwargs:
            assert [c.id for c in ours.output_axis] == kwargs["outputs"]
        if "inputs" in kwargs:
            assert [c.id for c in ours.input_axis] == kwargs["inputs"]
            assert np.asarray(ours.jacobian).shape == (1, len(kwargs["inputs"]))
    single = _ours_weighted(
        portfolio,
        request=PortfolioWeightedRiskRequest_(outputs=["trade:1:output:0"], weights=[5.0]),
    )
    assert single.component_means == [pytest.approx(TWO_SLOT_A_VALUE, rel=1e-9)]
    assert single.weighted_value == pytest.approx(5.0 * TWO_SLOT_A_VALUE, rel=1e-9)


def test_portfolio_weighted_jacobian_is_the_weighted_row_sum(dal):
    portfolio = _ours_portfolio([SIMPLE, TWO_SLOT], _ours_models(_MODEL, _MODEL2))
    outputs = ["trade:0:payoff", "trade:1:payoff"]
    weights = [3.0, 5.0]
    inputs = ["model:1:parameter:0", "model:0:parameter:1"]
    ours = _ours_weighted(
        portfolio,
        request=PortfolioWeightedRiskRequest_(outputs=outputs, weights=weights, inputs=inputs),
    )
    theirs = _theirs_weighted(
        dal,
        _native_portfolio(dal, [SIMPLE, TWO_SLOT], _native_models(dal, _MODEL, _MODEL2)),
        request=dal.PortfolioWeightedRiskRequest_(outputs=outputs, weights=weights, inputs=inputs),
    )
    _assert_weighted_parity(ours, theirs)
    per_output = _ours_jacobian(
        portfolio,
        request=PortfolioJacobianRiskRequest_(outputs=outputs, inputs=inputs),
    )
    rows = np.asarray(per_output.jacobian)
    np.testing.assert_allclose(
        np.asarray(ours.jacobian)[0], 3.0 * rows[0] + 5.0 * rows[1], rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(ours.component_means, per_output.values, rtol=1e-9)


def test_portfolio_weighted_shared_model_merges_groups(dal):
    model = BSModelData_New(*_MODEL)
    native_model = dal.BSModelData_New(*_MODEL)
    ours = _ours_weighted(_ours_portfolio([SIMPLE, SIMPLE], [model, model]))
    theirs = _theirs_weighted(
        dal, _native_portfolio(dal, [SIMPLE, SIMPLE], [native_model, native_model])
    )
    _assert_weighted_parity(ours, theirs)
    (group,) = ours.execution.groups
    assert group.trade_positions == [0, 1]
    assert group.evaluator_calls == 128
    assert group.generated_scenarios == 64


# --------------------------------------------------------------------------- #
# Validation, guards, ordering
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("selector", "error_class", "message"),
    [
        (
            ["model:0"],
            InvalidRiskRequest,
            "InvalidRiskRequest: unknown input ID; input=model:0",
        ),
        (
            ["model:0:parameter:0", "model:0:parameter:0"],
            InvalidRiskRequest,
            "InvalidRiskRequest: repeated input ID; input=model:0:parameter:0",
        ),
        (
            ["nonsense"],
            InvalidRiskRequest,
            "InvalidRiskRequest: unknown input ID; input=nonsense",
        ),
        (
            ["payoff"],
            InvalidPortfolioJacobianRequest,
            "InvalidPortfolioJacobianRequest: unknown scalar output ID; output=payoff",
        ),
        (
            ["trade:0:payoff", "TRADE:0:PAYOFF"],
            InvalidPortfolioJacobianRequest,
            "InvalidPortfolioJacobianRequest: repeated output ID; output=TRADE:0:PAYOFF",
        ),
        (
            [],
            InvalidPortfolioJacobianRequest,
            "InvalidPortfolioJacobianRequest: output selection must be nonempty; field=outputs",
        ),
    ],
)
def test_portfolio_jacobian_rejects_bad_selectors(dal, selector, error_class, message):
    portfolio = _ours_portfolio([SIMPLE, TWO_SLOT], _ours_models(_MODEL, _MODEL2))
    native_portfolio = _native_portfolio(
        dal, [SIMPLE, TWO_SLOT], _native_models(dal, _MODEL, _MODEL2)
    )
    field = "outputs" if error_class is InvalidPortfolioJacobianRequest else "inputs"
    with pytest.raises((RuntimeError, TypeError), match=re.escape(message.split(":")[0])):
        _theirs_jacobian(
            dal,
            native_portfolio,
            request=dal.PortfolioJacobianRiskRequest_(**{field: selector}),
        )
    with pytest.raises(error_class, match=re.escape(message)) as error:
        _ours_jacobian(portfolio, request=PortfolioJacobianRiskRequest_(**{field: selector}))
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("selector", "error_class", "message"),
    [
        (
            ["nonsense"],
            InvalidPortfolioWeightedRequest,
            "InvalidPortfolioWeightedRequest: unknown scalar output ID; output=nonsense",
        ),
        (
            ["trade:1:payoff", "trade:1:payoff"],
            InvalidPortfolioWeightedRequest,
            "InvalidPortfolioWeightedRequest: repeated output ID; output=trade:1:payoff",
        ),
        (
            [],
            InvalidPortfolioWeightedRequest,
            "InvalidPortfolioWeightedRequest: output selection must be nonempty; field=outputs",
        ),
        (
            ["model:0"],
            InvalidRiskRequest,
            "InvalidRiskRequest: unknown input ID; input=model:0",
        ),
    ],
)
def test_portfolio_weighted_rejects_bad_selectors(dal, selector, error_class, message):
    portfolio = _ours_portfolio([SIMPLE, TWO_SLOT], _ours_models(_MODEL, _MODEL2))
    native_portfolio = _native_portfolio(
        dal, [SIMPLE, TWO_SLOT], _native_models(dal, _MODEL, _MODEL2)
    )
    field = "outputs" if error_class is InvalidPortfolioWeightedRequest else "inputs"
    with pytest.raises(RuntimeError, match=re.escape(message.split(":")[0])):
        _theirs_weighted(
            dal,
            native_portfolio,
            request=dal.PortfolioWeightedRiskRequest_(**{field: selector}),
        )
    with pytest.raises(error_class, match=re.escape(message)) as error:
        _ours_weighted(portfolio, request=PortfolioWeightedRiskRequest_(**{field: selector}))
    assert str(error.value) == message


WEIGHT_ERRORS = [
    pytest.param(
        {"weights": [1.0]},
        "weight count must match selected outputs",
        "InvalidPortfolioWeightedRequest: weight count must match selected outputs; field=weights",
        id="count-default-outputs",
    ),
    pytest.param(
        {"weights": [1.0, 2.0], "outputs": ["trade:0:payoff"]},
        "weight count must match selected outputs",
        "InvalidPortfolioWeightedRequest: weight count must match selected outputs; field=weights",
        id="count-selected-outputs",
    ),
    pytest.param(
        {"weights": [float("nan"), 1.0]},
        "weight must be finite",
        "InvalidPortfolioWeightedRequest: weight must be finite; output=trade:0:payoff",
        id="nan-names-first-output",
    ),
    pytest.param(
        {"weights": [1.0, float("inf")]},
        "weight must be finite",
        "InvalidPortfolioWeightedRequest: weight must be finite; output=trade:1:payoff",
        id="inf-names-second-output",
    ),
]


@pytest.mark.parametrize(("kwargs", "native_match", "message"), WEIGHT_ERRORS)
def test_portfolio_weighted_validates_weights(dal, kwargs, native_match, message):
    portfolio = _ours_portfolio([SIMPLE, TWO_SLOT], _ours_models(_MODEL, _MODEL2))
    native_portfolio = _native_portfolio(
        dal, [SIMPLE, TWO_SLOT], _native_models(dal, _MODEL, _MODEL2)
    )
    with pytest.raises(RuntimeError, match=native_match):
        _theirs_weighted(dal, native_portfolio, request=dal.PortfolioWeightedRiskRequest_(**kwargs))
    with pytest.raises(InvalidPortfolioWeightedRequest, match=re.escape(message)) as error:
        _ours_weighted(portfolio, request=PortfolioWeightedRiskRequest_(**kwargs))
    assert str(error.value) == message


METADATA_ERRORS = [
    pytest.param(
        "jacobian",
        {"inputs": 0},
        "InvalidPortfolioRiskRequest: portfolio request; inputs; "
        "type=int, value=0; expected list or tuple, or None",
        id="inputs-int",
    ),
    pytest.param(
        "jacobian",
        {"inputs": ["model:0", 1]},
        "InvalidPortfolioRiskRequest: portfolio request; inputs; "
        "type=int, value=1; expected str or DAL String_ without NUL",
        id="inputs-element",
    ),
    pytest.param(
        "weighted",
        {"outputs": 0},
        "InvalidPortfolioRiskRequest: portfolio request; outputs; "
        "type=int, value=0; expected list or tuple, or None",
        id="outputs-int",
    ),
    pytest.param(
        "jacobian",
        {"report_factors": "x"},
        "InvalidPortfolioRiskRequest: portfolio request; report_factors; "
        "type=str, value='x'; expected list or tuple, or None",
        id="factors-string",
    ),
    pytest.param(
        "jacobian",
        {"report_factors": ["a"]},
        "InvalidPortfolioRiskRequest: portfolio request; report_factors; "
        "type=str, value='a'; expected int or float, excluding bool",
        id="factors-element",
    ),
    pytest.param(
        "weighted",
        {"report_factors": [True]},
        "InvalidPortfolioRiskRequest: portfolio request; report_factors; "
        "type=bool, value=True; expected int or float, excluding bool",
        id="factors-bool",
    ),
    pytest.param(
        "jacobian",
        {"numeric_payload_budget_bytes": -1},
        "InvalidPortfolioRiskRequest: portfolio request; numeric_payload_budget_bytes; "
        "type=int, value=-1; expected nonnegative size_t integer or None, excluding bool",
        id="numeric-budget",
    ),
    pytest.param(
        "weighted",
        {"recording_capacity_budget_bytes": "x"},
        "InvalidPortfolioRiskRequest: portfolio request; recording_capacity_budget_bytes; "
        "type=str, value='x'; expected nonnegative size_t integer or None, excluding bool",
        id="recording-budget",
    ),
    pytest.param(
        "weighted",
        {"scratch_capacity_budget_bytes": -1},
        "InvalidPortfolioRiskRequest: portfolio request; scratch_capacity_budget_bytes; "
        "type=int, value=-1; expected nonnegative size_t integer or None, excluding bool",
        id="scratch-budget",
    ),
    pytest.param(
        "jacobian",
        {"max_block_width": 1.5},
        "InvalidPortfolioRiskRequest: PortfolioJacobianRiskRequest_; max_block_width; "
        "type=float, value=1.5; expected nonnegative size_t integer or None, excluding bool",
        id="width-type",
    ),
    pytest.param(
        "jacobian",
        {"max_block_width": True},
        "InvalidPortfolioRiskRequest: PortfolioJacobianRiskRequest_; max_block_width; "
        "type=bool, value=True; expected nonnegative size_t integer or None, excluding bool",
        id="width-bool",
    ),
    pytest.param(
        "jacobian",
        {"max_block_width": 32769},
        "InvalidPortfolioRiskRequest: max_block_width must be in [1, 32768]",
        id="width-range",
    ),
    pytest.param(
        "weighted",
        {"weights": 0},
        "InvalidPortfolioRiskRequest: PortfolioWeightedRiskRequest_; weights; "
        "type=int, value=0; expected list or tuple, or None",
        id="weights-int",
    ),
    pytest.param(
        "weighted",
        {"weights": ["a"]},
        "InvalidPortfolioRiskRequest: PortfolioWeightedRiskRequest_; weights; "
        "type=str, value='a'; expected int or float, excluding bool",
        id="weights-element",
    ),
    pytest.param(
        "weighted",
        {"weights": [True]},
        "InvalidPortfolioRiskRequest: PortfolioWeightedRiskRequest_; weights; "
        "type=bool, value=True; expected int or float, excluding bool",
        id="weights-bool",
    ),
]


def _portfolio_request(variant, **kwargs):
    cls = PortfolioJacobianRiskRequest_ if variant == "jacobian" else PortfolioWeightedRiskRequest_
    return cls(**kwargs)


def _native_portfolio_request(dal, variant, **kwargs):
    cls = (
        dal.PortfolioJacobianRiskRequest_
        if variant == "jacobian"
        else dal.PortfolioWeightedRiskRequest_
    )
    return cls(**kwargs)


@pytest.mark.parametrize(("variant", "kwargs", "message"), METADATA_ERRORS)
def test_portfolio_requests_reject_bad_metadata(dal, variant, kwargs, message):
    #  Native raises at construction (pybind TypeError / core RuntimeError),
    #  ours at the valuation-time boundary.
    portfolio = _ours_portfolio([SIMPLE, SIMPLE], _ours_models(_MODEL, _MODEL2))
    with pytest.raises((RuntimeError, TypeError), match=re.escape(message)):
        _native_portfolio_request(dal, variant, **kwargs)
    value = _ours_jacobian if variant == "jacobian" else _ours_weighted
    with pytest.raises(InvalidPortfolioRiskRequest, match=re.escape(message)) as error:
        value(portfolio, request=_portfolio_request(variant, **kwargs))
    assert str(error.value) == message


def test_portfolio_requests_accept_metadata(dal):
    #  Large-enough budgets and all-ones factors change nothing observable.
    portfolio = _ours_portfolio([SIMPLE, SIMPLE], _ours_models(_MODEL, _MODEL2))
    native_portfolio = _native_portfolio(
        dal, [SIMPLE, SIMPLE], _native_models(dal, _MODEL, _MODEL2)
    )
    kwargs = {
        "report_factors": [1.0, 1, 1.0, 1, 1.0, 1, 1.0, 1],
        "numeric_payload_budget_bytes": 2**32,
        "recording_capacity_budget_bytes": 2**32,
        "scratch_capacity_budget_bytes": 2**32,
    }
    ours = _ours_jacobian(portfolio, request=PortfolioJacobianRiskRequest_(**kwargs))
    theirs = _theirs_jacobian(
        dal, native_portfolio, request=dal.PortfolioJacobianRiskRequest_(**kwargs)
    )
    _assert_jacobian_parity(ours, theirs)
    assert ours.execution.requested_max_block_width == 1
    wide = _ours_jacobian(portfolio, request=PortfolioJacobianRiskRequest_(max_block_width=4))
    _assert_jacobian_parity(
        wide,
        _theirs_jacobian(
            dal, native_portfolio, request=dal.PortfolioJacobianRiskRequest_(max_block_width=4)
        ),
    )
    assert wide.execution.requested_max_block_width == 4


@pytest.mark.parametrize(
    ("n_paths", "entry", "suffix"),
    [
        (
            0,
            PortfolioMonteCarlo_ValueWithJacobianRisk,
            "; number of Monte Carlo paths must be positive",
        ),
        (
            2**31,
            PortfolioMonteCarlo_ValueWithJacobianRisk,
            "; number of Monte Carlo paths must be positive",
        ),
        (1.5, PortfolioMonteCarlo_ValueWithJacobianRisk, ""),
        (True, PortfolioMonteCarlo_ValueWithJacobianRisk, ""),
        (
            0,
            PortfolioMonteCarlo_ValueWithWeightedRisk,
            "; number of Monte Carlo paths must be positive",
        ),
    ],
)
def test_portfolio_validates_path_counts(dal, n_paths, entry, suffix):
    function = entry.__name__
    portfolio = _ours_portfolio([SIMPLE, SIMPLE], _ours_models(_MODEL, _MODEL2))
    message = (
        f"InvalidPathCount: {function}; num_path / numPath; "
        f"type={type(n_paths).__name__}, value={n_paths!r}; "
        f"expected a positive integer in 1..2147483647{suffix}"
    )
    native_portfolio = _native_portfolio(
        dal, [SIMPLE, SIMPLE], _native_models(dal, _MODEL, _MODEL2)
    )
    with pytest.raises((RuntimeError, TypeError), match="InvalidPathCount"):
        getattr(dal, function)(native_portfolio, n_paths, valuation=_native_valuation(dal))
    with pytest.raises(InvalidPathCount, match=re.escape(message)) as error:
        entry(portfolio, n_paths, valuation=_VALUATION)
    assert str(error.value) == message


REQUEST_TYPE_ERRORS = [
    pytest.param(
        PortfolioMonteCarlo_ValueWithJacobianRisk,
        "RiskRequest_",
        lambda: RiskRequest_(),
        "PortfolioJacobianRiskRequest_",
        id="risk-to-jacobian",
    ),
    pytest.param(
        PortfolioMonteCarlo_ValueWithJacobianRisk,
        "JacobianRiskRequest_",
        JacobianRiskRequest_,
        "PortfolioJacobianRiskRequest_",
        id="per-trade-jacobian-to-portfolio",
    ),
    pytest.param(
        PortfolioMonteCarlo_ValueWithWeightedRisk,
        "RiskRequest_",
        lambda: RiskRequest_(),
        "PortfolioWeightedRiskRequest_",
        id="risk-to-weighted",
    ),
    pytest.param(
        PortfolioMonteCarlo_ValueWithWeightedRisk,
        "PortfolioJacobianRiskRequest_",
        PortfolioJacobianRiskRequest_,
        "PortfolioWeightedRiskRequest_",
        id="jacobian-to-weighted",
    ),
]


@pytest.mark.parametrize(("entry", "name", "make_request", "expected"), REQUEST_TYPE_ERRORS)
def test_portfolio_enforce_request_types(dal, entry, name, make_request, expected):
    portfolio = _ours_portfolio([SIMPLE, SIMPLE], _ours_models(_MODEL, _MODEL2))
    request = make_request()
    message = f"InvalidSetting: {entry.__name__}; request; type={name}; expected {expected} or None"
    native_portfolio = _native_portfolio(
        dal, [SIMPLE, SIMPLE], _native_models(dal, _MODEL, _MODEL2)
    )
    with pytest.raises(TypeError, match="InvalidSetting"):
        getattr(dal, entry.__name__)(
            native_portfolio, 64, valuation=_native_valuation(dal), request=request
        )
    with pytest.raises(InvalidSetting, match=re.escape(message)) as error:
        entry(portfolio, 64, valuation=_VALUATION, request=request)
    assert str(error.value) == message


def test_portfolio_path_count_precedes_request_type(dal):
    portfolio = _ours_portfolio([SIMPLE, SIMPLE], _ours_models(_MODEL, _MODEL2))
    native_portfolio = _native_portfolio(
        dal, [SIMPLE, SIMPLE], _native_models(dal, _MODEL, _MODEL2)
    )
    with pytest.raises(RuntimeError, match="InvalidPathCount"):
        dal.PortfolioMonteCarlo_ValueWithJacobianRisk(
            native_portfolio, 0, valuation=_native_valuation(dal), request=dal.RiskRequest_()
        )
    with pytest.raises(InvalidPathCount, match="InvalidPathCount"):
        PortfolioMonteCarlo_ValueWithJacobianRisk(
            portfolio, 0, valuation=_VALUATION, request=RiskRequest_()
        )


_GUARD = "UnsupportedPortfolioRisk: exercise and fully expired trades are unsupported"
_RECEIVERLESS = (
    "InvalidScriptPortfolio: trade axes cannot be indexed; field=products; trade=t1; "
    "cause=product must contain a PAYS or EXERCISE payoff"
)


def test_portfolio_guards_exercise_and_expired_trades(dal):
    exercise = ([(2024, 1, 2), (2024, 7, 2)], ["x = 1\nexercise x", "y PAYS SPOT()"])
    expired = ([(2022, 1, 15)], ["p pays 1.0"])
    value_ours = _ours_jacobian
    value_theirs = _theirs_jacobian
    for dates, events, specs in (
        (None, None, [exercise, SIMPLE]),
        (None, None, [expired, SIMPLE]),
        (None, None, [SIMPLE, expired]),
    ):
        ours = _ours_portfolio(specs, _ours_models(_MODEL, _MODEL2))
        theirs = _native_portfolio(dal, specs, _native_models(dal, _MODEL, _MODEL2))
        with pytest.raises(RuntimeError, match="UnsupportedPortfolioRisk"):
            value_theirs(dal, theirs)
        with pytest.raises(UnsupportedPortfolioRisk, match=re.escape(_GUARD)) as error:
            value_ours(ours)
        assert str(error.value) == _GUARD
    #  The guard fires even when the offending trade's outputs are unselected.
    exercise_portfolio = _ours_portfolio([exercise, SIMPLE], _ours_models(_MODEL, _MODEL2))
    with pytest.raises(UnsupportedPortfolioRisk, match=re.escape(_GUARD)):
        _ours_jacobian(
            exercise_portfolio,
            request=PortfolioJacobianRiskRequest_(outputs=["trade:1:payoff"]),
        )
    #  Weighted variant rejects the same trades.
    with pytest.raises(UnsupportedPortfolioRisk, match=re.escape(_GUARD)):
        _ours_weighted(exercise_portfolio)
    #  Partly expired products stay live, as at the per-trade level.
    partly = ([(2022, 1, 15), (2024, 1, 2)], ["a = 1", "a pays spot()"])
    ours = _ours_jacobian(_ours_portfolio([partly, SIMPLE], _ours_models(_MODEL, _MODEL2)))
    theirs = _theirs_jacobian(
        dal, _native_portfolio(dal, [partly, SIMPLE], _native_models(dal, _MODEL, _MODEL2))
    )
    _assert_jacobian_parity(ours, theirs)


def test_portfolio_receiverless_trades_fail_at_axis_indexing(dal):
    vector_only = ([(2024, 1, 2)], ["APPEND(v,SPOT())"])
    empty = ([], [])
    scalar_only = ([(2024, 1, 2)], ["p = 1"])
    #  Native's IndexTradeAxes makes its own has-PAYS-or-EXERCISE check (the
    #  "must contain" cause) before delegating to the risk-output axis.
    for spec in (vector_only, empty, scalar_only):
        ours = _ours_portfolio([spec, SIMPLE], _ours_models(_MODEL, _MODEL2))
        theirs = _native_portfolio(dal, [spec, SIMPLE], _native_models(dal, _MODEL, _MODEL2))
        with pytest.raises(RuntimeError, match="product must contain a PAYS or EXERCISE payoff"):
            _theirs_jacobian(dal, theirs)
        with pytest.raises(InvalidScriptPortfolio, match=re.escape(_RECEIVERLESS)) as error:
            _ours_jacobian(ours)
        assert str(error.value) == _RECEIVERLESS
    #  Axis indexing precedes selector validation (both selectors), as in native.
    for selector in ({"inputs": ["nonsense"]}, {"outputs": ["nonsense"]}):
        with pytest.raises(InvalidScriptPortfolio, match=re.escape(_RECEIVERLESS)):
            _ours_jacobian(
                _ours_portfolio([scalar_only, SIMPLE], _ours_models(_MODEL, _MODEL2)),
                request=PortfolioJacobianRiskRequest_(**selector),
            )
    with pytest.raises(InvalidScriptPortfolio, match=re.escape(_RECEIVERLESS)):
        _ours_weighted(_ours_portfolio([scalar_only, SIMPLE], _ours_models(_MODEL, _MODEL2)))


def test_portfolio_exercise_without_pays_delegates_receiverless_cause(dal):
    #  EXERCISE without PAYS passes the axis guard's own check but the delegated
    #  risk-output axis rejects it; the cause is the delegated message verbatim.
    exercise_only = ([(2024, 1, 2)], ["s = 1\nexercise s"])
    message = (
        "InvalidScriptPortfolio: trade axes cannot be indexed; field=products; trade=t1; "
        "cause=InvalidWeightedRiskRequest: indexed scalar default receiver is required; "
        "output=payoff"
    )
    ours = _ours_portfolio([exercise_only, SIMPLE], _ours_models(_MODEL, _MODEL2))
    theirs = _native_portfolio(dal, [exercise_only, SIMPLE], _native_models(dal, _MODEL, _MODEL2))
    with pytest.raises(RuntimeError, match="indexed scalar default receiver is required"):
        _theirs_jacobian(dal, theirs)
    with pytest.raises(InvalidScriptPortfolio, match=re.escape(message)) as error:
        _ours_jacobian(ours)
    assert str(error.value) == message
    with pytest.raises(RuntimeError, match="indexed scalar default receiver is required"):
        _theirs_weighted(dal, theirs)
    with pytest.raises(InvalidScriptPortfolio, match=re.escape(message)):
        _ours_weighted(ours)
    #  Still an axis-time failure: it wins over a bad input selector.
    with pytest.raises(InvalidScriptPortfolio, match=re.escape(message)):
        _ours_jacobian(ours, request=PortfolioJacobianRiskRequest_(inputs=["nonsense"]))


def test_script_portfolio_length_precedes_element_checks(dal):
    #  The equal-length check precedes the per-element product/model checks,
    #  while trade_ids' own element checks precede the length check.
    ours_products = _ours_products([SIMPLE] * 3)
    theirs_products = _native_products(dal, [SIMPLE] * 3)
    ours_models = _ours_models(_MODEL, _MODEL, _MODEL)
    theirs_models = _native_models(dal, _MODEL, _MODEL, _MODEL)
    length_error = (
        "InvalidScriptPortfolio: trade_ids, products and modelData must have equal nonzero lengths"
    )
    #  Each case pairs unequal lengths with a bad in-range element on one side.
    cases = [
        (
            ["t1", "t2"],
            theirs_products,
            [theirs_models[0], 42],
            ours_products,
            [ours_models[0], 42],
        ),
        (
            ["t1", "t2"],
            [theirs_products[0], 42, theirs_products[2]],
            theirs_models[:2],
            [ours_products[0], 42, ours_products[2]],
            ours_models[:2],
        ),
        (
            ["t1", "t2"],
            [theirs_products[0], 42],
            theirs_models,
            [ours_products[0], 42],
            ours_models,
        ),
    ]
    for ids, theirs_prods, theirs_mods, ours_prods, ours_mods in cases:
        with pytest.raises(RuntimeError, match="equal nonzero lengths"):
            dal.ScriptPortfolio_New(ids, theirs_prods, theirs_mods)
        with pytest.raises(InvalidScriptPortfolio, match=re.escape(length_error)) as error:
            ScriptPortfolio_New(ids, ours_prods, ours_mods)
        assert str(error.value) == length_error
    id_error = (
        "InvalidScriptPortfolio: ScriptPortfolio_New; trade_ids; "
        "type=int, value=2; expected str or DAL String_ without NUL"
    )
    with pytest.raises(TypeError, match=re.escape(id_error)):
        dal.ScriptPortfolio_New(["t1", 2], theirs_products, theirs_models)
    with pytest.raises(InvalidScriptPortfolio, match=re.escape(id_error)) as error:
        ScriptPortfolio_New(["t1", 2], ours_products, ours_models)
    assert str(error.value) == id_error


@pytest.mark.parametrize(
    ("portfolio_request", "message"),
    [
        (
            PortfolioJacobianRiskRequest_(inputs=["nonsense"], outputs=["nonsense"]),
            "InvalidPortfolioJacobianRequest: unknown scalar output ID; output=nonsense",
        ),
        (
            PortfolioJacobianRiskRequest_(inputs=["nonsense"]),
            "InvalidRiskRequest: unknown input ID; input=nonsense",
        ),
    ],
    ids=["outputs-before-inputs", "selectors-before-guard"],
)
def test_portfolio_jacobian_validation_order(dal, portfolio_request, message):
    exercise = ([(2024, 1, 2), (2024, 7, 2)], ["x = 1\nexercise x", "y PAYS SPOT()"])
    #  Selector errors shadow the exercise guard, which runs after them.
    guarded = _ours_portfolio([exercise, SIMPLE], _ours_models(_MODEL, _MODEL2))
    native_guarded = _native_portfolio(
        dal, [exercise, SIMPLE], _native_models(dal, _MODEL, _MODEL2)
    )
    with pytest.raises(RuntimeError, match=re.escape(message.split(":")[0])):
        native_kwargs = {
            field: list(getattr(portfolio_request, field))
            for field in ("inputs", "outputs")
            if getattr(portfolio_request, field) is not None
        }
        _theirs_jacobian(
            dal,
            native_guarded,
            request=dal.PortfolioJacobianRiskRequest_(**native_kwargs),
        )
    error_class = (
        InvalidPortfolioJacobianRequest
        if message.startswith("InvalidPortfolioJacobianRequest")
        else InvalidRiskRequest
    )
    with pytest.raises(error_class, match=re.escape(message)) as error:
        _ours_jacobian(guarded, request=portfolio_request)
    assert str(error.value) == message


def test_portfolio_factor_count_follows_input_selection(dal):
    #  The factor count check runs after input selection, naming the base code.
    portfolio = _ours_portfolio([SIMPLE, SIMPLE], _ours_models(_MODEL, _MODEL2))
    native_portfolio = _native_portfolio(
        dal, [SIMPLE, SIMPLE], _native_models(dal, _MODEL, _MODEL2)
    )
    message = "InvalidRiskRequest: report factor count must match selected inputs"
    for factors in ([2.0], [2.0] * 20):
        with pytest.raises(RuntimeError, match="report factor count"):
            _theirs_jacobian(
                dal,
                native_portfolio,
                request=dal.PortfolioJacobianRiskRequest_(report_factors=factors),
            )
        with pytest.raises(InvalidRiskRequest, match=re.escape(message)) as error:
            _ours_jacobian(portfolio, request=PortfolioJacobianRiskRequest_(report_factors=factors))
        assert str(error.value) == message
    #  Matching the selected inputs keeps the scaled reported jacobian.
    ours = _ours_jacobian(
        portfolio,
        request=PortfolioJacobianRiskRequest_(inputs=["model:0:parameter:0"], report_factors=[2.0]),
    )
    theirs = _theirs_jacobian(
        dal,
        native_portfolio,
        request=dal.PortfolioJacobianRiskRequest_(
            inputs=["model:0:parameter:0"], report_factors=[2.0]
        ),
    )
    _assert_jacobian_parity(ours, theirs)
    assert np.asarray(ours.reported_jacobian)[0, 0] == pytest.approx(
        2.0 * np.asarray(ours.jacobian)[0, 0]
    )
