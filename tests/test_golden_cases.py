"""Golden cases: end-to-end expectations over the real app, seeded from jsonl.

The same file doubles as the offline eval set cited in DESIGN.md.
"""

import json
from pathlib import Path

import pytest
from test_api_stream import fast_app, post_stream, text_of, types_of

CASES_PATH = Path(__file__).parent / "cases" / "basic.jsonl"
CASES = [json.loads(line) for line in CASES_PATH.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("case", CASES, ids=[case["message"] for case in CASES])
async def test_golden_case(case):
    events = await post_stream(fast_app(), {"message": case["message"]})

    tool_events = [e for e in events if e["type"].startswith("tool_")]
    if case["expected_tool"] is None:
        assert tool_events == []
    else:
        assert tool_events, "expected a tool call"
        assert all(e["tool_name"] == case["expected_tool"] for e in tool_events)

    assert types_of(events)[-1] == "message_done"

    text = text_of(events)
    for needle in case["must_include"]:
        assert needle in text, f"missing {needle!r} in {text!r}"
    for needle in case["must_not_include"]:
        assert needle not in text, f"unexpected {needle!r} in {text!r}"
