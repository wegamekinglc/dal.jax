"""Example contracts must be authored with the script engine."""

import ast
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _reference(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.alias):
        return node.name.rsplit(".", 1)[-1]
    return None


@pytest.mark.parametrize("source", sorted(EXAMPLES.glob("*.py")), ids=lambda source: source.name)
def test_examples_do_not_construct_handwritten_payoffs(source):
    references = {_reference(node) for node in ast.walk(ast.parse(source.read_text()))}
    forbidden = references & {"PathProduct", "MonteCarloEngine"}
    assert not forbidden, (
        f"{source.name}: use prepare(...).engine(...) instead of {sorted(forbidden)}"
    )
