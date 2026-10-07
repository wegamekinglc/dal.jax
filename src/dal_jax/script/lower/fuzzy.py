"""DAL fuzzy conditions and recursive IF blending, sharing safe scalar arithmetic."""

import jax.numpy as jnp
import numpy as np

from dal_jax.script import ast as A
from dal_jax.script.lower import smoothing as S
from dal_jax.script.lower.exact import _Lowerer
from dal_jax.script.passes.intervals import EPSILON

_KERNELS = {
    "dal": (S.cspr, S.cspr_bounds, S.bfly, S.bfly_bounds),
    "smoothstep": (S.smoothstep_cspr, S.smoothstep_cspr_bounds, S.smoothstep_bfly, S.smoothstep_bfly_bounds),
}


class _FuzzyLowerer(_Lowerer):
    def __init__(self, const_names: tuple[str, ...], smooth: float, kernel: str) -> None:
        super().__init__(const_names, historical=False, backend=jnp)
        self.smooth, self.kernels = smooth, _KERNELS[kernel]
        self._expressions.update({
            A.And: lambda n: self._binary(n, lambda a, b: a * b),
            A.Or: lambda n: self._binary(n, lambda a, b: a + b - a * b),
            A.Not: lambda n: self._unary(n, lambda a: 1.0 - a),
            A.TrueNode: lambda n: lambda state, sample, params, active: 1.0,
            A.FalseNode: lambda n: lambda state, sample, params, active: 0.0,
        })

    def expression(self, node: A.Node):
        if isinstance(node, A.Comparison):
            return self._comparison(node)
        return super().expression(node)

    def _comparison(self, node: A.Comparison):
        operand = self.expression(node.args[0])
        start = 2 if isinstance(node, A.Equal) else 0
        continuous, discrete = self.kernels[start:start + 2]
        kernel = (lambda x: discrete(x, node.lb, node.rb)) if node.is_discrete else (
            lambda x: continuous(x, self.smooth if node.eps < 0 else node.eps))
        return lambda state, sample, params, active: kernel(operand(state, sample, params, active))

    def _if(self, node: A.If):
        if isinstance(node.condition, (A.TrueNode, A.FalseNode)):
            return self.event(node.then_branch if isinstance(node.condition, A.TrueNode) else node.else_branch)
        condition = self.expression(node.condition)
        then, otherwise = self.event(node.then_branch), self.event(node.else_branch)
        indices = np.asarray(node.affected_vars, dtype=np.int32)

        def evaluate(state, sample, params, active):
            degree = condition(state, sample, params, active)
            # 1-EPSILON rounds to 1 in float32; equality must still mask
            # the unused branch before its unsafe arithmetic is evaluated.
            full = degree >= 1.0 if jnp.result_type(degree) == jnp.float32 else degree > 1.0 - EPSILON
            zero = degree < EPSILON
            left = then(state, sample, params, jnp.logical_and(active, jnp.logical_not(zero)))
            right = otherwise(state, sample, params, jnp.logical_and(active, jnp.logical_not(full)))
            # Both branch inputs are safe at the endpoints; selecting explicitly
            # also drops the condition's adjoint when DAL executes only one side.
            blend = degree * left[indices] + (1.0 - degree) * right[indices]
            values = jnp.where(full, left[indices], jnp.where(zero, right[indices], blend))
            return self._write(state, indices, values)

        return evaluate


def lower_event(event: A.Event, const_names: tuple[str, ...] = (), *, smooth: float = 0.01, kernel: str = "dal"):
    """Fuzzy ``event(state, sample, script_params) -> state``; historical replay stays exact."""
    return _FuzzyLowerer(const_names, smooth, kernel).event(event)
