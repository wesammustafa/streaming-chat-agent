import pytest

from app.tools.calculator import (
    MAX_EXPRESSION_LENGTH,
    CalculatorTool,
    format_number,
    validate_expression,
)

ALLOWED = [
    ("2+2", "4"),
    ("2 + 2", "4"),
    ("10/4", "2.5"),
    ("10 / 2", "5"),
    ("-5 + 3", "-2"),
    ("3 - -2", "5"),
    ("+7", "7"),
    ("(2 + 3) * 4", "20"),
    ("2.5 * 2", "5"),
    ("1.5 + 1.25", "2.75"),
    ("1000000 * 1000000", "1000000000000"),
]

DENIED = [
    "__import__('os').system('id')",
    "().__class__",
    "abs(1)",
    "a + 1",
    "2 ** 10",
    "1 // 2",
    "1 % 2",
    "1 < 2",
    "[1, 2]",
    '"a" + "b"',
    "True + 1",
    "(x := 1)",
    "lambda: 1",
    "",
    "   ",
    "1 +",
    "9" * (MAX_EXPRESSION_LENGTH + 1),
]


@pytest.mark.parametrize(("expression", "expected"), ALLOWED)
async def test_allowed_expressions_evaluate(expression, expected):
    result = await CalculatorTool().run(expression)
    assert result.ok, result.error
    assert result.content == expected


@pytest.mark.parametrize("expression", DENIED)
async def test_denied_expressions_fail_closed(expression):
    result = await CalculatorTool().run(expression)
    assert not result.ok
    assert result.error


async def test_division_by_zero_is_a_clean_error():
    result = await CalculatorTool().run("1 / 0")
    assert not result.ok
    assert result.error == "division by zero"


def test_validate_expression_agrees_with_tool():
    for expression, _ in ALLOWED:
        assert validate_expression(expression) is None
    for expression in DENIED:
        assert validate_expression(expression) is not None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(5.0, "5"), (2.5, "2.5"), (4, "4"), (-3.0, "-3"), (0.1, "0.1")],
)
def test_format_number(value, expected):
    assert format_number(value) == expected
