# Streaming Chat Agent

A minimal chat assistant that streams its replies over NDJSON and can call a
calculator tool mid-conversation. FastAPI backend, dependency-free HTML/JS
frontend, deterministic rule-based model, fully offline test suite.

The design goal: the smallest version that exercises the real architecture of
a streaming, tool-calling assistant. Component boundaries, error paths, and
the streaming contract are production-shaped; heavy infrastructure is
deliberately deferred. See [DESIGN.md](DESIGN.md).

## Run

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run uvicorn app.main:app
```

Open http://127.0.0.1:8000 and chat.

Without uv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn
uvicorn app.main:app
```

## Try these prompts

- `what is (10 - 4) * 2?` calls the calculator and streams the answer
- `1 / 0` demonstrates the tool error path; the reply still completes cleanly
- `hello!` plain chat, no tool involved

The pill above the assistant's reply shows live tool status: running, result,
or error.

## Test

```bash
uv run pytest
```

97 tests in five layers (calculator, planner, service orchestration, HTTP API,
golden cases), all offline with zero injected delays; the whole suite runs in
well under two seconds. Lint with `uv run ruff check .`

## API

`POST /api/chat/stream` with `{"message": "...", "conversation_id": "..."}`.
`conversation_id` is optional; the server generates one and returns it in the
first event. The response is `application/x-ndjson`, one event per line:

```text
{"type":"message_start","conversation_id":"..."}
{"type":"tool_start","tool_name":"calculator"}
{"type":"tool_result","tool_name":"calculator","result":"12"}
{"type":"text_delta","text":"(10 - 4) "}
{"type":"text_delta","text":"* "}
{"type":"text_delta","text":"2 "}
{"type":"text_delta","text":"= "}
{"type":"text_delta","text":"12"}
{"type":"message_done","conversation_id":"..."}
```

Every stream ends in exactly one terminal event: `message_done` or `error`.
Invalid input (empty or oversized message, oversized conversation id) gets a
400 before any streaming starts.

## Layout

```text
app/
  main.py       composition root (create_app) and static mount
  api/          HTTP layer: validation, streaming response, terminal-event wrapper
  domain/       messages, actions, stream events, tool contracts
  services/     assistant orchestration, tool registry, conversation store
  models/       AssistantModel protocol + rule-based implementation (all copy lives here)
  tools/        AST-validated calculator
  static/       index.html + app.js
tests/          five test layers + golden cases (tests/cases/basic.jsonl)
```
