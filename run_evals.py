"""Thin eval CLI over the golden cases in tests/cases/basic.jsonl.

Replays each case against the app in-process (no server, no network for the
deterministic default) and prints one line per case plus a summary. The model
comes from ASSISTANT_MODEL, so the same gate can be run by hand against
ollama: ASSISTANT_MODEL=ollama uv run python run_evals.py
Exit code 0 means every case passed.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from app.main import create_app
from app.models.rule_based import RuleBasedAssistantModel

DEFAULT_CASES = Path(__file__).parent / "tests" / "cases" / "basic.jsonl"


def build_app() -> FastAPI:
    choice = os.environ.get("ASSISTANT_MODEL", "deterministic").strip().lower()
    if choice == "deterministic":
        # Production wiring minus the UI pacing delay; evals need no theatrics.
        return create_app(model=RuleBasedAssistantModel(chunk_delay_seconds=0))
    return create_app()


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def run_case(client: httpx.AsyncClient, case: dict[str, Any]) -> list[str]:
    """Return the case's failure reasons; empty means it passed."""
    events: list[dict[str, Any]] = []
    payload = {"message": case["message"]}
    async with client.stream("POST", "/api/chat/stream", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.strip():
                events.append(json.loads(line))

    failures: list[str] = []
    tool_events = [e for e in events if e["type"].startswith("tool_")]
    seen_tools = sorted({e["tool_name"] for e in tool_events})
    expected_tool = case["expected_tool"]
    if expected_tool is None:
        if tool_events:
            failures.append(f"expected no tool, saw {seen_tools}")
    elif not tool_events:
        failures.append(f"expected tool {expected_tool}, saw none")
    elif seen_tools != [expected_tool]:
        failures.append(f"expected only {expected_tool}, saw {seen_tools}")

    if not events or events[-1]["type"] != "message_done":
        failures.append("stream did not end in message_done")

    text = "".join(e["text"] for e in events if e["type"] == "text_delta")
    for needle in case["must_include"]:
        if needle not in text:
            failures.append(f"missing {needle!r} in {text!r}")
    for needle in case["must_not_include"]:
        if needle in text:
            failures.append(f"unexpected {needle!r} in {text!r}")
    return failures


async def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the golden cases against the app.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="path to a jsonl file")
    args = parser.parse_args()
    cases = load_cases(args.cases)

    transport = httpx.ASGITransport(app=build_app())
    passed = 0
    # No client timeout: a local LLM warming up can pause long past any sane default.
    client = httpx.AsyncClient(transport=transport, base_url="http://evals", timeout=None)
    async with client:
        for case in cases:
            failures = await run_case(client, case)
            if not failures:
                passed += 1
            print(f"{'PASS' if not failures else 'FAIL'}  {case['message']}")
            for reason in failures:
                print(f"      {reason}")

    print(f"\n{passed}/{len(cases)} golden cases passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
