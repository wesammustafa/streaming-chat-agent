import json

import httpx

from app.main import create_app


async def test_stream_endpoint_returns_parseable_ndjson_with_terminal_event():
    transport = httpx.ASGITransport(app=create_app())
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", "/api/chat/stream", json={"message": "hello"}) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        lines = [line async for line in response.aiter_lines() if line.strip()]

    events = [json.loads(line) for line in lines]
    assert events[0]["type"] == "message_start"
    assert events[-1]["type"] == "message_done"
