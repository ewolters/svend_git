"""Safe formula evaluation for PCL calculated measures.

Uses [slug] syntax to reference other measures. Evaluated via restricted
AST walking — same pattern as hoshin/hoshin_calculations.py but with
slug resolution instead of {{fieldname}} variables.

SECURITY NOTE: This module does NOT use eval(). It parses formulas into an
AST and walks only whitelisted node types (arithmetic ops, constants, names,
safe function calls). No arbitrary code execution is possible.
"""

import ast
import math
import operator
import re

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "pow": pow,
}

_SLUG_PATTERN = re.compile(r"\[([a-zA-Z0-9_-]+)\]")
_MAX_LEN = 500
_MAX_NODES = 100
_MAX_DEPTH = 20


def extract_slugs(formula: str) -> list[str]:
    """Extract [slug] references from a formula. Returns deduplicated list."""
    return list(dict.fromkeys(_SLUG_PATTERN.findall(formula)))


def _normalize(formula: str) -> str:
    """Replace [slug] with bare names for AST parsing."""

    def _replace(m):
        return m.group(1).replace("-", "_")

    return _SLUG_PATTERN.sub(_replace, formula)


def _count_nodes(node) -> int:
    c = 1
    for child in ast.iter_child_nodes(node):
        c += _count_nodes(child)
    return c


def _max_depth(node, depth=0) -> int:
    if depth > _MAX_DEPTH:
        return depth
    child_depths = [_max_depth(child, depth + 1) for child in ast.iter_child_nodes(node)]
    return max(child_depths) if child_depths else depth


def evaluate_pcl_formula(formula: str, variables: dict[str, float]) -> float:
    """Safely evaluate a PCL formula with [slug] references.

    Uses restricted AST walking — only whitelisted arithmetic operations,
    numeric constants, variable lookups, and safe math functions are allowed.
    No exec/eval, no imports, no attribute access.

    Args:
        formula: e.g. "[avail] * [perf] * [qual]"
        variables: dict mapping slug to float values.

    Returns:
        float result

    Raises:
        ValueError: if formula is unsafe, too complex, or references unknown slugs
        ZeroDivisionError: if formula divides by zero
    """
    if len(formula) > _MAX_LEN:
        raise ValueError(f"Formula too long (max {_MAX_LEN} chars)")

    normalized = _normalize(formula)
    norm_vars = {k.replace("-", "_"): v for k, v in variables.items()}

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid formula syntax: {e}")

    if _count_nodes(tree) > _MAX_NODES:
        raise ValueError("Formula too complex (too many nodes)")

    if _max_depth(tree) > _MAX_DEPTH:
        raise ValueError("Formula too deep (max nesting exceeded)")

    def _walk(node):
        if isinstance(node, ast.Expression):
            return _walk(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        if isinstance(node, ast.Name):
            name = node.id
            if name in norm_vars:
                return norm_vars[name]
            if name in _SAFE_FUNCS:
                return _SAFE_FUNCS[name]
            raise ValueError(f"Unknown variable: [{name.replace('_', '-')}]")

        if isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(_walk(node.left), _walk(node.right))

        if isinstance(node, ast.UnaryOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(_walk(node.operand))

        if isinstance(node, ast.Call):
            func = _walk(node.func)
            if func not in _SAFE_FUNCS.values():
                raise ValueError("Unsupported function call")
            args = [_walk(arg) for arg in node.args]
            return func(*args)

        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    return _walk(tree)
