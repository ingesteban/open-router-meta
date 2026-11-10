"""Adapter tests: OpenRouter HTTP mocked with respx."""

import httpx
import pytest
import respx

from app.domain.entities import ImagePart, Message, Role
from app.domain.exceptions import LLMAuthError, LLMRateLimitError
from app.infrastructure.openrouter_client import OpenRouterClient

BASE = "https://openrouter.ai/api/v1"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@respx.mock
async def test_complete_sends_multimodal_payload_and_parses_usage():
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "meta/muse-spark-1.3",
                "choices": [{"message": {"role": "assistant", "content": "A cake"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": 0.0001},
            },
        )
    )
    client = OpenRouterClient("sk-test", app_title="t")
    msg = Message(Role.USER, "what is this?", images=(ImagePart("https://x/y.png"),))
    out = await client.complete([msg], model="meta/muse-spark-1.3")

    assert out.text == "A cake" and out.usage.total_tokens == 12 and out.usage.cost == 0.0001
    sent = route.calls[0].request
    assert sent.headers["Authorization"] == "Bearer sk-test"
    body = httpx.Response(200, content=sent.content).json()
    assert body["messages"][0]["content"][1] == {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
    await client.aclose()


@pytest.mark.anyio
@respx.mock
async def test_stream_yields_deltas_and_skips_comments():
    sse = (
        ": OPENROUTER PROCESSING\n\n"
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, text=sse))
    client = OpenRouterClient("sk-test")
    tokens = [t async for t in client.stream([Message(Role.USER, "hi")], model="m")]
    assert "".join(tokens) == "Hello"
    await client.aclose()


@pytest.mark.anyio
@respx.mock
async def test_error_mapping():
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(402, json={"error": {"message": "Insufficient credits"}})
    )
    client = OpenRouterClient("sk-test")
    with pytest.raises(LLMRateLimitError, match="Insufficient credits"):
        await client.complete([Message(Role.USER, "hi")], model="m")
    await client.aclose()


@pytest.mark.anyio
async def test_missing_key_raises_auth_error():
    client = OpenRouterClient("")
    with pytest.raises(LLMAuthError):
        await client.complete([Message(Role.USER, "hi")], model="m")
    await client.aclose()
