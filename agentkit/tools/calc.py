"""A safe arithmetic evaluator.

Kept separate from the MCP server so the pure logic is unit-testable without spawning a server.
Only numeric literals and a fixed set of arithmetic operators are allowed - no names, calls or
attribute access - so it is safe to run on model-provided input.
"""

from __future__ import annotations

import ast
import operator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalcError(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise CalcError(f"unsupported expression element: {ast.dump(node)}")


def safe_eval(expression: str) -> float:
    """Evaluate a basic arithmetic expression, raising CalcError on anything unsupported."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"invalid expression: {expression!r}") from exc
    return _eval(tree.body)
