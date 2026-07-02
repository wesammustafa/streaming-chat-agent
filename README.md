# Streaming Chat Agent

A minimal chat assistant that streams its replies over NDJSON and can call
tools mid-conversation: an AST-validated calculator and a weather lookup.
FastAPI backend, dependency-free HTML/JS frontend, deterministic rule-based
model by default with an optional local-LLM mode, fully offline test suite.

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
pip install fastapi uvicorn httpx
uvicorn app.main:app
```

## Model modes

`ASSISTANT_MODEL` selects the assistant model at startup.

### deterministic (default)

The rule-based planner plus the calculator tool. Fully offline and
reproducible; this is what the test suite and CI always run against. No setup
needed, no env vars required.

### ollama (optional, local LLM)

Routes planning and replies through a local [Ollama](https://ollama.com)
server for realistic multilingual chat. The LLM's only planning power is
choosing between a `calculator` call, a `weather_lookup` call, or a direct
reply; every proposed plan is validated before execution and anything invalid
falls back to a direct reply. Recommended model: Qwen2.5 7B Instruct.

```bash
ollama pull qwen2.5:7b
ASSISTANT_MODEL=ollama uv run uvicorn app.main:app
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `ASSISTANT_MODEL` | `deterministic` | `deterministic` or `ollama` |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model tag to use |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server address |
| `WEATHER_SOURCE` | `fixture` | `fixture` (ten canned cities) or `live` (Open-Meteo) |

Try: `what's the weather in Madrid?` or `¿qué tiempo hace en Lisboa?`

`WEATHER_SOURCE=live` swaps the fixture for real current weather from
[Open-Meteo](https://open-meteo.com) (free, no API key), for any city its
geocoder knows. Privacy note: in live mode the place name the planner extracts
from the user's message is sent to Open-Meteo; the default fixture keeps
everything on your machine.

```bash
ASSISTANT_MODEL=ollama WEATHER_SOURCE=live uv run uvicorn app.main:app
```

Both weather tools answer to the same `weather_lookup` name, so validated
plans route the same way whichever is wired. The planner's tool menu is built
from the registered tools' specs. The fixture remains the default; tests never
touch the network.

Notes: Ollama mode exists for local demo realism, not production reliability.
It is non-deterministic and excluded from the test suite and CI, which always
use the deterministic model. The weather data itself is local, fake, and
deterministic. No API keys, no paid services. If the Ollama server is not
running, the assistant replies with setup guidance instead of failing.

## Try these prompts

- `what is (10 - 4) * 2?` calls the calculator and streams the answer
- `1 / 0` demonstrates the tool error path; the reply still completes cleanly
- `hello!` plain chat, no tool involved

The pill above the assistant's reply shows live tool status: running, result,
or error. While a reply streams, the send button becomes a stop button;
stopping keeps the partial text on screen and drops it from history.

## Test

```bash
uv run pytest
```

The suite spans calculator, planner, service orchestration, HTTP API, golden
cases, tools, and model selection, all offline with zero injected delays; it
runs in well under two seconds and never requires Ollama. CI runs the same
checks on every push and pull request. Lint with `uv run ruff check .` and
type-check with `uv run mypy`.

## Evals

`uv run python run_evals.py` replays the golden cases from
`tests/cases/basic.jsonl` against the app in-process and prints a per-case
pass/fail summary (non-zero exit on any failure). pytest already covers the
same cases; the CLI exists to gate a different model by hand:

```bash
ASSISTANT_MODEL=ollama uv run python run_evals.py
```

## API

`POST /api/chat/stream` with `{"message": "...", "conversation_id": "..."}`.
`conversation_id` is optional; the server generates one and returns it in the
first event. The response is `application/x-ndjson`, one event per line.
Here `what's the weather in Lisbon?` under the ollama model; the wire format
is identical for every model and tool:

```text
{"type":"message_start","conversation_id":"..."}
{"type":"tool_start","tool_name":"weather_lookup"}
{"type":"tool_result","tool_name":"weather_lookup","result":"Lisbon: 24°C, breezy"}
{"type":"text_delta","text":"Right "}
{"type":"text_delta","text":"now "}
{"type":"text_delta","text":"it's "}
{"type":"text_delta","text":"24°C "}
{"type":"text_delta","text":"and breezy "}
{"type":"text_delta","text":"in Lisbon."}
{"type":"message_done","conversation_id":"..."}
```

Every stream ends in exactly one terminal event: `message_done` or `error`.
Invalid input (empty or oversized message, malformed conversation id; ids are
`[A-Za-z0-9_-]`, max 64 chars) gets a 400 before any streaming starts.

## Layout

```text
app/
  main.py       composition root (create_app) and static mount
  api/          HTTP layer: validation, streaming response, terminal-event wrapper
  domain/       messages, actions, stream events, tool contracts
  services/     assistant orchestration, tool registry, conversation store
  models/       AssistantModel protocol + rule-based default + optional Ollama adapter
  tools/        AST-validated calculator, fixture + live Open-Meteo weather lookups
  static/       index.html + app.js
tests/          offline unit, service, API, adapter, and golden-case layers
run_evals.py    golden-case eval CLI (see Evals)
.github/        CI: ruff, mypy, pytest on every push and pull request
```
