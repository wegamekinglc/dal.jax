"""Port of DAL's ``tests/script/test_domain.cpp`` (bounds, intervals, domains)."""

import pytest

from dal_jax.errors import ScriptError
from dal_jax.script.passes.intervals import MINUS_INF, PLUS_INF, Bound, Domain, Interval

B = Bound
I = Interval.of  # noqa: E741


def S(x):
    return Interval.singleton(x)


def test_bound_accessors_and_infinities():
    b = B(3.5)
    assert not b.is_infinite
    assert not b.plus_inf
    assert not b.minus_inf
    assert b.real == 3.5
    assert B().is_zero
    assert PLUS_INF.is_infinite
    assert PLUS_INF.plus_inf
    assert PLUS_INF.is_positive()
    assert PLUS_INF.is_positive(True)
    assert not PLUS_INF.is_zero
    assert MINUS_INF.minus_inf
    assert MINUS_INF.is_negative()
    assert MINUS_INF.is_negative(True)
    assert not MINUS_INF.is_zero


def test_bound_zero_sign_semantics():
    z = B(0.0)
    assert z.is_positive(False) and not z.is_positive(True)
    assert z.is_negative(False) and not z.is_negative(True)
    assert z.is_zero


def test_bound_comparisons():
    assert B(1.0) == B(1.0)
    assert B(1.0) != B(2.0)
    assert B(1.0) < B(2.0)
    assert B(2.0) > B(1.0)
    assert B(2.0) <= B(2.0)
    assert B(2.0) >= B(2.0)
    assert not B(2.0) < B(2.0)
    assert MINUS_INF < B(0.0) < PLUS_INF
    assert MINUS_INF < PLUS_INF
    assert PLUS_INF > MINUS_INF
    assert PLUS_INF == B(1e29, plus_inf=True)
    assert MINUS_INF == B(-1e29, minus_inf=True)


def test_bound_negation_and_multiplication():
    assert -B(3.0) == B(-3.0) and (-PLUS_INF).minus_inf and (-MINUS_INF).plus_inf
    assert B(2.0) * B(3.0) == B(6.0) and B(-2.0) * B(3.0) == B(-6.0)
    assert (PLUS_INF * B(2.0)).plus_inf and (PLUS_INF * B(-2.0)).minus_inf
    assert (MINUS_INF * B(2.0)).minus_inf and (MINUS_INF * B(-2.0)).plus_inf
    assert (PLUS_INF * MINUS_INF).minus_inf and (PLUS_INF * PLUS_INF).plus_inf
    assert (PLUS_INF * B(0.0)).plus_inf and (B(0.0) * MINUS_INF).minus_inf


def test_interval_kinds():
    i = S(4.0)
    assert i.singleton_value() == 4.0 and not i.is_continuous and i.left == B(4.0) and not i.is_infinite
    assert S(0.0).is_zero and S(0.0).is_singleton
    c = I(B(1.0), B(2.0))
    assert not c.is_singleton and c.is_continuous and not c.is_zero


@pytest.mark.parametrize("left,right", [(B(2.0), B(1.0)), (PLUS_INF, B(1.0)), (B(1.0), MINUS_INF)])
def test_interval_inconsistent_bounds(left, right):
    with pytest.raises(ScriptError):
        I(left, right)


def test_interval_signs():
    pos, neg, straddle = I(B(1.0), B(2.0)), I(B(-2.0), B(-1.0)), I(B(-1.0), B(1.0))
    assert pos.is_positive()
    assert pos.is_positive(True)
    assert not pos.is_negative()
    assert pos.is_pos_or_neg()
    assert neg.is_negative()
    assert neg.is_negative(True)
    assert not neg.is_positive()
    assert not straddle.is_positive(True)
    assert not straddle.is_negative(True)
    assert not straddle.is_pos_or_neg(True)
    real = I(MINUS_INF, PLUS_INF)
    assert real.is_infinite
    assert real.includes(0.0)
    assert real.includes(1e20)


def test_interval_arithmetic():
    a, b = I(B(1.0), B(2.0)), I(B(3.0), B(4.0))
    c = a + b
    assert c.left == B(4.0) and c.right == B(6.0)
    d = a + I(MINUS_INF, B(5.0))
    assert d.left.minus_inf and d.right == B(7.0)
    e = I(B(5.0), B(7.0)) - I(B(1.0), B(2.0))
    assert e.left == B(3.0) and e.right == B(6.0)
    n = -I(B(1.0), B(3.0))
    assert n.left == B(-3.0) and n.right == B(-1.0)
    m = I(B(2.0), B(3.0)) * I(B(4.0), B(5.0))
    assert m.left == B(8.0) and m.right == B(15.0)
    s = I(B(-2.0), B(3.0)) * I(B(-1.0), B(4.0))
    assert s.left == B(-8.0) and s.right == B(12.0)
    assert (I(B(2.0), B(3.0)) * S(0.0)).is_zero


def test_interval_inverse_and_division():
    assert S(4.0).inverse().singleton_value() == pytest.approx(0.25)
    inv = I(B(2.0), B(4.0)).inverse()
    assert inv.left.real == pytest.approx(0.25) and inv.right.real == pytest.approx(0.5)
    with pytest.raises(ScriptError):
        S(0.0).inverse()
    q = I(B(2.0), B(4.0)) / I(B(2.0), B(2.0))
    assert q.left.real == pytest.approx(1.0) and q.right.real == pytest.approx(2.0)


def test_interval_min_max_includes_order():
    a, b = I(B(1.0), B(4.0)), I(B(2.0), B(3.0))
    mn, mx = a.imin(b), a.imax(b)
    assert (mn.left, mn.right) == (B(1.0), B(3.0))
    assert (mx.left, mx.right) == (B(2.0), B(4.0))
    big, inner = I(B(1.0), B(5.0)), I(B(2.0), B(4.0))
    assert big.includes(3.0)
    assert big.includes(1.0)
    assert big.includes(5.0)
    assert not big.includes(6.0)
    assert not big.includes(0.0)
    assert big.includes_interval(inner)
    assert not inner.includes_interval(big)
    x, y, z = I(B(1.0), B(2.0)), I(B(1.0), B(3.0)), I(B(2.0), B(3.0))
    assert x.same(x)
    assert x.less(y)
    assert y.less(z)
    assert not z.less(x)


def test_interval_intersect_and_merge():
    a, b, c = I(B(1.0), B(4.0)), I(B(3.0), B(6.0)), I(B(10.0), B(12.0))
    assert a.intersects(b) and not a.intersects(c)
    merged = a.merged(b)
    assert merged.left == B(1.0) and merged.right == B(6.0)


def test_domain_empty_and_singleton():
    d = Domain()
    assert d.is_empty and len(d.intervals) == 0 and d.min_bound().minus_inf and d.max_bound().plus_inf
    s = Domain.value(3.0)
    assert not s.is_empty and len(s.intervals) == 1 and s.is_discrete and s.constant_value() == 3.0
    assert s.min_bound() == B(3.0) and s.max_bound() == B(3.0)


def test_domain_insertion():
    d = Domain(S(1.0), S(3.0))
    assert len(d.intervals) == 2 and d.is_discrete and d.singletons() == [1.0, 3.0]
    merged = Domain(I(B(1.0), B(4.0)), I(B(3.0), B(6.0)))
    assert len(merged.intervals) == 1 and merged.min_bound() == B(1.0) and merged.max_bound() == B(6.0)
    disjoint = Domain(I(B(1.0), B(2.0)), I(B(5.0), B(6.0)))
    assert len(disjoint.intervals) == 2
    real = Domain(I(B(1.0), B(2.0)), I(MINUS_INF, PLUS_INF))
    assert len(real.intervals) == 1 and real.is_infinite and real.min_bound().minus_inf and real.max_bound().plus_inf


def test_domain_queries():
    assert Domain(S(0.0), S(1.0)).boolean_values() == (0.0, 1.0)
    pos = Domain(I(B(1.0), B(2.0)), S(5.0))
    assert pos.is_positive() and not pos.is_negative()
    neg = Domain(I(B(-3.0), B(-1.0)))
    assert neg.is_negative() and not neg.is_positive()
    cont = Domain(S(0.0), I(B(2.0), B(4.0))).continuous()
    assert len(cont.intervals) == 1 and not cont.is_discrete and cont.min_bound() == B(2.0)


def test_domain_arithmetic():
    assert (Domain.value(2.0) + Domain.value(3.0)).constant_value() == pytest.approx(5.0)
    n = -Domain(I(B(1.0), B(3.0)))
    assert n.min_bound() == B(-3.0) and n.max_bound() == B(-1.0)
    assert (Domain.value(5.0) - Domain.value(2.0)).constant_value() == pytest.approx(3.0)
    assert (Domain.value(4.0) * Domain.value(3.0)).constant_value() == pytest.approx(12.0)
    shifted = Domain(I(B(1.0), B(2.0))).shifted(3.0)
    assert shifted.min_bound() == B(4.0) and shifted.max_bound() == B(5.0)
    back = shifted.shifted(-1.0)
    assert back.min_bound() == B(3.0) and back.max_bound() == B(4.0)
    a, b = Domain(I(B(1.0), B(4.0))), Domain(I(B(2.0), B(3.0)))
    assert (a.dmin(b).min_bound(), a.dmin(b).max_bound()) == (B(1.0), B(3.0))
    assert (a.dmax(b).min_bound(), a.dmax(b).max_bound()) == (B(2.0), B(4.0))


def test_domain_inclusion_and_zero_shortcuts():
    d = Domain(I(B(1.0), B(3.0)), S(7.0))
    assert d.includes(2.0)
    assert d.includes(7.0)
    assert not d.includes(5.0)
    zero = Domain.value(0.0)
    assert zero.can_be_zero()
    assert not zero.can_be_non_zero()
    assert zero.zero_is_discrete()
    assert not zero.zero_is_cont()
    around = Domain(I(B(-1.0), B(1.0)))
    assert around.can_be_zero()
    assert around.can_be_non_zero()
    assert not around.zero_is_discrete()
    assert around.zero_is_cont()
    nonzero = Domain.value(5.0)
    assert not nonzero.can_be_zero()
    assert nonzero.can_be_non_zero()


def test_domain_sign_possibilities_and_nearest_bounds():
    pos = Domain(I(B(1.0), B(2.0)))
    assert pos.can_be_positive(True) and not pos.can_be_negative(True)
    straddle = Domain(I(B(-2.0), B(3.0)))
    assert straddle.can_be_positive(True) and straddle.can_be_negative(True)
    assert not Domain().can_be_positive(True) and not Domain().can_be_negative(True)
    assert Domain(I(B(-2.0), B(-1.0)), I(B(1.0), B(3.0))).smallest_pos_lb() == pytest.approx(1.0)
    assert Domain(I(B(-3.0), B(-1.0)), I(B(1.0), B(2.0))).biggest_neg_rb() == pytest.approx(-1.0)


def test_domain_add_domain():
    a = Domain.value(1.0)
    a.add_domain(Domain.value(2.0))
    assert len(a.intervals) == 2 and a.includes(1.0) and a.includes(2.0)
