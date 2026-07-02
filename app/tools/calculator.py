"""Safe arithmetic over a whitelisted Python AST.

validate_expression is exported so the planner can pre-validate candidate
expressions with the exact same rules the tool enforces.
"""

import ast

from app.domain.tools import ToolResult, ToolSpec

MAX_EXPRESSION_LENGTH = 100

_ALLOWED_STRUCTURE = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.UAdd,
    ast.USub,
)


def validate_expression(text: str) -> str | None:
    """Return why text is not a safe arithmetic expression, or None if it is."""
    if not text.strip():
        return "expression is empty"
    if len(text) > MAX_EXPRESSION_LENGTH:
        return f"expression longer than {MAX_EXPRESSION_LENGTH} characters"
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return "not a valid arithmetic expression"
    for node in ast.walk(tree):
        if isinstance(node, _ALLOWED_STRUCTURE):
            continue
        # bool subclasses int, so type() rather than isinstance keeps True/False out.
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            continue
        return "only numbers, parentheses, and + - * / are supported"
    return None


def _evaluate(node: ast.AST) -> int | float:
    match node:
        case ast.Expression(body=body):
            return _evaluate(body)
        case ast.Constant(value=value):
            return value
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return +_evaluate(operand)
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_evaluate(operand)
        case ast.BinOp(left=left, op=op, right=right):
            lhs, rhs = _evaluate(left), _evaluate(right)
            match op:
                case ast.Add():
                    return lhs + rhs
                case ast.Sub():
                    return lhs - rhs
                case ast.Mult():
                    return lhs * rhs
                case ast.Div():
                    return lhs / rhs
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def format_number(value: int | float) -> str:
    """Integral floats render as ints (5.0 -> "5"); division always yields floats."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class CalculatorTool:
    spec = ToolSpec(
        name="calculator",
        description="Evaluates arithmetic with numbers, parentheses, and + - * /.",
    )

    async def run(self, tool_input: str) -> ToolResult:
        error = validate_expression(tool_input)
        if error is not None:
            return ToolResult.failed(error)
        try:
            value = _evaluate(ast.parse(tool_input, mode="eval"))
        except ZeroDivisionError:
            return ToolResult.failed("division by zero")
        return ToolResult.succeeded(format_number(value))
