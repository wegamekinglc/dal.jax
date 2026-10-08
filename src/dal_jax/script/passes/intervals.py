"""Interval arithmetic of DAL's ``visitor/domain.hpp``: ``Bound_``, ``Interval_``, ``Domain_``.

Comparisons carry DAL's tolerance ``EPSILON = 2e-14`` and infinite bounds are
flagged (their value is ``INF = 1e29``).  A domain is a sorted set of disjoint
intervals; inserting an interval merges every interval it touches, and an
element equivalent to an existing one is dropped, as ``std::set`` does.
"""

import math
from dataclasses import dataclass

from dal_jax.errors import script_error

EPSILON = 2e-14
INF = 1e29


@dataclass(frozen=True, slots=True)
class Bound:
    real: float = 0.0
    plus_inf: bool = False
    minus_inf: bool = False

    @property
    def is_infinite(self) -> bool:
        return self.plus_inf or self.minus_inf

    def is_positive(self, strict: bool = False) -> bool:
        return self.plus_inf or self.real > (EPSILON if strict else -EPSILON)

    def is_negative(self, strict: bool = False) -> bool:
        return self.minus_inf or self.real < (-EPSILON if strict else EPSILON)

    @property
    def is_zero(self) -> bool:
        return not self.is_infinite and abs(self.real) < EPSILON

    def __eq__(self, rhs: object) -> bool:
        if not isinstance(rhs, Bound):
            return NotImplemented
        return (
            (self.plus_inf and rhs.plus_inf)
            or (self.minus_inf and rhs.minus_inf)
            or abs(self.real - rhs.real) < EPSILON
        )

    def __hash__(self) -> int:  # pragma: no cover - fuzzy equality, never hashed
        return 0

    def __lt__(self, rhs: "Bound") -> bool:
        return (
            (self.minus_inf and not rhs.minus_inf)
            or (not self.plus_inf and rhs.plus_inf)
            or self.real < rhs.real - EPSILON
        )

    def __gt__(self, rhs: "Bound") -> bool:
        return (
            (not self.minus_inf and rhs.minus_inf)
            or (self.plus_inf and not rhs.plus_inf)
            or self.real > rhs.real + EPSILON
        )

    def __le__(self, rhs: "Bound") -> bool:
        return not self > rhs

    def __ge__(self, rhs: "Bound") -> bool:
        return not self < rhs

    def _same_strict_sign(self, rhs: "Bound") -> bool:
        return (self.is_positive(True) and rhs.is_positive(True)) or (
            self.is_negative(True) and rhs.is_negative(True)
        )

    def __mul__(self, rhs: "Bound") -> "Bound":
        if not (self.is_infinite or rhs.is_infinite):
            return Bound(self.real * rhs.real)
        if self._same_strict_sign(rhs):
            return PLUS_INF
        if self.is_zero:
            return rhs  # 0 * inf = inf here, as in DAL
        return self if rhs.is_zero else MINUS_INF

    def __neg__(self) -> "Bound":
        if self.minus_inf:
            return PLUS_INF
        if self.plus_inf:
            return MINUS_INF
        return Bound(-self.real)


PLUS_INF = Bound(INF, plus_inf=True)
MINUS_INF = Bound(-INF, minus_inf=True)


def _min_bound(bounds: list[Bound]) -> Bound:
    result = bounds[0]
    for b in bounds[1:]:
        if b < result:
            result = b
    return result


def _max_bound(bounds: list[Bound]) -> Bound:
    result = bounds[0]
    for b in bounds[1:]:
        if result < b:
            result = b
    return result


@dataclass(frozen=True, slots=True, eq=False)
class Interval:
    left: Bound
    right: Bound

    @staticmethod
    def singleton(value: float) -> "Interval":
        return Interval(Bound(value), Bound(value))

    @staticmethod
    def of(left: Bound, right: Bound) -> "Interval":
        if left == PLUS_INF or right == MINUS_INF or not left <= right:
            raise script_error("Inconsistent bounds")
        return Interval(left, right)

    def is_positive(self, strict: bool = False) -> bool:
        return self.left.is_positive(strict)

    def is_negative(self, strict: bool = False) -> bool:
        return self.right.is_negative(strict)

    def is_pos_or_neg(self, strict: bool = False) -> bool:
        return self.is_positive(strict) or self.is_negative(strict)

    @property
    def is_infinite(self) -> bool:
        return self.left.is_infinite or self.right.is_infinite

    def singleton_value(self) -> float | None:
        return self.left.real if not self.is_infinite and self.left == self.right else None

    @property
    def is_singleton(self) -> bool:
        return self.singleton_value() is not None

    @property
    def is_zero(self) -> bool:
        return self.is_singleton and self.left.is_zero

    @property
    def is_continuous(self) -> bool:
        return not self.is_singleton

    def same(self, rhs: "Interval") -> bool:
        return self.left == rhs.left and self.right == rhs.right

    def less(self, rhs: "Interval") -> bool:
        return self.left < rhs.left or (self.left == rhs.left and self.right < rhs.right)

    def __add__(self, rhs: "Interval") -> "Interval":
        lb = (
            MINUS_INF
            if self.left.minus_inf or rhs.left.minus_inf
            else Bound(self.left.real + rhs.left.real)
        )
        rb = (
            PLUS_INF
            if self.right.plus_inf or rhs.right.plus_inf
            else Bound(self.right.real + rhs.right.real)
        )
        return Interval.of(lb, rb)

    def __neg__(self) -> "Interval":
        return Interval.of(-self.right, -self.left)

    def __sub__(self, rhs: "Interval") -> "Interval":
        return self + -rhs

    def __mul__(self, rhs: "Interval") -> "Interval":
        if self.is_zero or rhs.is_zero:
            return Interval.singleton(0.0)
        b = [
            self.right * rhs.right,
            self.right * rhs.left,
            self.left * rhs.right,
            self.left * rhs.left,
        ]
        return Interval.of(_min_bound(b), _max_bound(b))

    def _inverse_away_from_zero(self) -> "Interval":
        """1/x for an interval of one strict sign."""
        if not self.is_infinite:
            return Interval.of(Bound(1.0 / self.right.real), Bound(1.0 / self.left.real))
        if self.is_positive():
            return Interval.of(Bound(0.0), Bound(1.0 / self.left.real))
        return Interval.of(Bound(1.0 / self.right.real), Bound(0.0))

    def _inverse_touching_zero(self) -> "Interval":
        """1/x for an interval with 0 as one of its bounds."""
        if self.is_infinite:
            return (
                Interval.of(Bound(0.0), PLUS_INF)
                if self.is_positive()
                else Interval.of(MINUS_INF, Bound(0.0))
            )
        if self.is_positive():
            return Interval.of(Bound(1.0 / self.right.real), PLUS_INF)
        return Interval.of(MINUS_INF, Bound(1.0 / self.left.real))

    def inverse(self) -> "Interval":
        if self.is_zero:
            raise script_error("Division by {0}")
        if (v := self.singleton_value()) is not None:
            return Interval.singleton(_c_div(1.0, v))
        if self.is_pos_or_neg(True):
            return self._inverse_away_from_zero()
        if self.left.is_zero or self.right.is_zero:
            return self._inverse_touching_zero()
        return Interval.of(MINUS_INF, PLUS_INF)

    def __truediv__(self, rhs: "Interval") -> "Interval":
        return self * rhs.inverse()

    def imin(self, rhs: "Interval") -> "Interval":
        lb = rhs.left if rhs.left < self.left else self.left
        rb = rhs.right if rhs.right < self.right else self.right
        return Interval.of(lb, rb)

    def imax(self, rhs: "Interval") -> "Interval":
        lb = rhs.left if rhs.left > self.left else self.left
        rb = rhs.right if rhs.right > self.right else self.right
        return Interval.of(lb, rb)

    def includes(self, x: float) -> bool:
        return self.left <= Bound(x) and self.right >= Bound(x)

    def includes_interval(self, rhs: "Interval") -> bool:
        return self.left <= rhs.left and self.right >= rhs.right

    def intersects(self, rhs: "Interval") -> bool:
        lb = rhs.left if rhs.left > self.left else self.left
        rb = rhs.right if rhs.right < self.right else self.right
        return rb >= lb

    def merged(self, rhs: "Interval") -> "Interval":
        """``*this`` widened to cover ``rhs`` (DAL's in-place ``Merge``)."""
        left = rhs.left if rhs.left < self.left else self.left
        right = rhs.right if rhs.right > self.right else self.right
        return Interval(left, right)


REAL_LINE = Interval(MINUS_INF, PLUS_INF)
POSITIVE_HALF_LINE = Interval(Bound(0.0), PLUS_INF)


def _c_div(x: float, y: float) -> float:
    if y == 0.0:
        return math.copysign(math.inf, x) if x != 0.0 else math.nan
    return x / y


def c_log(x: float) -> float:
    if math.isnan(x) or x < 0.0:
        return math.nan
    return -math.inf if x == 0.0 else math.log(x)


def c_sqrt(x: float) -> float:
    return math.nan if math.isnan(x) or x < 0.0 else math.sqrt(x)


def c_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def c_pow(x: float, y: float) -> float:
    """C ``pow``: poles give signed infinities, invalid operations NaN, overflow infinity."""
    odd_integer = y.is_integer() and int(y) % 2 == 1
    try:
        return math.pow(x, y)
    except ValueError:
        if x == 0.0:
            return math.copysign(math.inf, x) if odd_integer else math.inf
        return math.nan
    except OverflowError:
        return -math.inf if x < 0.0 and odd_integer else math.inf


class Domain:
    """A set of disjoint intervals, sorted by ``Interval.less``."""

    __slots__ = ("intervals",)

    def __init__(self, *intervals: Interval) -> None:
        self.intervals: list[Interval] = []
        for interval in intervals:
            self.add_interval(interval)

    @staticmethod
    def value(x: float) -> "Domain":
        return Domain(Interval.singleton(x))

    @staticmethod
    def real() -> "Domain":
        return Domain(REAL_LINE)

    def copy(self) -> "Domain":
        result = Domain()
        result.intervals = list(self.intervals)
        return result

    def _insert(self, interval: Interval) -> None:
        for i, existing in enumerate(self.intervals):
            if interval.less(existing):
                self.intervals.insert(i, interval)
                return
            if not existing.less(interval):
                return  # equivalent: std::set keeps the existing element
        self.intervals.append(interval)

    def _lower_bound(self, interval: Interval) -> int:
        for i, existing in enumerate(self.intervals):
            if not existing.less(interval):
                return i
        return len(self.intervals)

    def _overlapping(self, interval: Interval) -> int | None:
        """Position of an existing interval that touches ``interval`` (DAL's ``lower_bound`` walk)."""
        left, right = interval.left, interval.right
        if self.intervals[0].left > right or self.intervals[-1].right < left:
            return None
        found = self._lower_bound(interval)
        if found == len(self.intervals) or self.intervals[found].left > right:
            found -= 1
        return None if self.intervals[found].right < left else found

    def add_interval(self, interval: Interval) -> None:
        if interval.left.minus_inf and interval.right.plus_inf and self.intervals:
            self.intervals = [REAL_LINE]
            return
        while self.intervals and (found := self._overlapping(interval)) is not None:
            interval = interval.merged(self.intervals.pop(found))
        self._insert(interval)

    def add_domain(self, rhs: "Domain") -> None:
        for interval in list(rhs.intervals):
            self.add_interval(interval)

    def is_positive(self, strict: bool = False) -> bool:
        return all(i.is_positive(strict) for i in self.intervals)

    @property
    def is_empty(self) -> bool:
        return not self.intervals

    @property
    def is_infinite(self) -> bool:
        return any(i.is_infinite for i in self.intervals)

    def min_bound(self) -> Bound:
        return self.intervals[0].left if self.intervals else MINUS_INF

    def max_bound(self) -> Bound:
        return self.intervals[-1].right if self.intervals else PLUS_INF

    def continuous(self) -> "Domain":
        return Domain(*(i for i in self.intervals if i.is_continuous))

    def boolean_values(self) -> tuple[float, float] | None:
        values = self.singletons()
        return (values[0], values[1]) if len(values) == 2 else None

    def shifted(self, x: float) -> "Domain":
        """``operator+=(x)``: every interval moved by ``x`` (equivalent results collapse)."""
        if abs(x) < EPSILON:
            return self.copy()
        result = Domain()
        for i in self.intervals:
            result._insert(i + Interval.singleton(x))
        return result

    def zero_is_cont(self) -> bool:
        return any(i.is_continuous and i.includes(0.0) for i in self.intervals)

    def is_negative(self, strict: bool = False) -> bool:
        return all(i.is_negative(strict) for i in self.intervals)

    @property
    def is_discrete(self) -> bool:
        return all(i.is_singleton for i in self.intervals)

    def singletons(self, discrete_only: bool = True) -> list[float]:
        values = []
        for interval in self.intervals:
            v = interval.singleton_value()
            if v is None:
                if discrete_only:
                    return []
            else:
                values.append(v)
        return values

    def constant_value(self) -> float | None:
        values = self.singletons()
        return values[0] if len(values) == 1 else None

    @property
    def is_constant(self) -> bool:
        return len(self.singletons()) == 1

    def _combine(self, rhs: "Domain", op) -> "Domain":
        result = Domain()
        for i in self.intervals:
            for j in rhs.intervals:
                result.add_interval(op(i, j))
        return result

    def __add__(self, rhs: "Domain") -> "Domain":
        return self._combine(rhs, Interval.__add__)

    def __sub__(self, rhs: "Domain") -> "Domain":
        return self._combine(rhs, Interval.__sub__)

    def __mul__(self, rhs: "Domain") -> "Domain":
        return self._combine(rhs, Interval.__mul__)

    def __truediv__(self, rhs: "Domain") -> "Domain":
        return self._combine(rhs, Interval.__truediv__)

    def __neg__(self) -> "Domain":
        result = Domain()
        for i in self.intervals:
            result.add_interval(-i)
        return result

    def dmin(self, rhs: "Domain") -> "Domain":
        return self._combine(rhs, Interval.imin)

    def dmax(self, rhs: "Domain") -> "Domain":
        return self._combine(rhs, Interval.imax)

    def apply(self, func, func_domain: Interval) -> "Domain":
        values = self.singletons()
        if not values:
            return Domain(func_domain)
        result = Domain()
        for v in values:
            result.add_interval(Interval.singleton(func(v)))
        return result

    def apply2(self, func, rhs: "Domain", func_domain: Interval) -> "Domain":
        values1, values2 = self.singletons(), rhs.singletons()
        if not values1 or not values2:
            return Domain(func_domain)
        result = Domain()
        for v1 in values1:
            for v2 in values2:
                result.add_interval(Interval.singleton(func(v1, v2)))
        return result

    def includes(self, x: float) -> bool:
        return any(i.includes(x) for i in self.intervals)

    def can_be_zero(self) -> bool:
        return self.includes(0.0)

    def can_be_non_zero(self) -> bool:
        return not (not self.intervals or (len(self.intervals) == 1 and self.intervals[0].is_zero))

    def zero_is_discrete(self) -> bool:
        return any(i.is_zero for i in self.intervals)

    def can_be_positive(self, strict: bool) -> bool:
        return bool(self.intervals) and self.intervals[-1].right.real > (
            EPSILON if strict else -EPSILON
        )

    def can_be_negative(self, strict: bool) -> bool:
        return bool(self.intervals) and self.intervals[0].left.real < (
            -EPSILON if strict else EPSILON
        )

    def smallest_pos_lb(self, strict: bool = False) -> float | None:
        if self.intervals[-1].left.is_negative(not strict):
            return None
        for interval in self.intervals:
            if not interval.left.is_negative(not strict):
                return interval.left.real
        return None

    def biggest_neg_rb(self, strict: bool = False) -> float | None:
        if self.intervals[0].right.is_positive(not strict):
            return None
        for interval in reversed(self.intervals):
            if not interval.right.is_positive(not strict):
                return interval.right.real
        return None

    def __repr__(self) -> str:
        def bound(b: Bound) -> str:
            return "+INF" if b.plus_inf else "-INF" if b.minus_inf else f"{b.real:g}"

        parts = [
            f"{{{i.left.real:g}}}" if i.is_singleton else f"({bound(i.left)},{bound(i.right)})"
            for i in self.intervals
        ]
        return "{" + ";".join(parts) + "}"
