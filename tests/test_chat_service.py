"""Use-case tests with a fake LLMClient (no network)."""

import pytest

from app.application.chat_service import ChatService
from app.domain.entities import Completion, Message, ModelInfo, Role
from app.domain.exceptions import InvalidRequestError, LLMProviderError


class FakeLLM:
    def __init__(self, reply: str = "hi") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def complete(self, messages, *, model, temperature=None, max_tokens=None, json_mode=False):
        self.calls.append({"messages": list(messages), "model": model, "json_mode": json_mode})
        return Completion(text=self.reply, model=model)

    async def stream(self, messages, *, model, temperature=None, max_tokens=None):
        for t in ["a", "b"]:
            yield t

    async def list_models(self):
        return [ModelInfo(id="openai/x", name="x"), ModelInfo(id="meta/b", name="b"), ModelInfo(id="meta/a", name="a")]


@pytest.mark.anyio
async def test_chat_uses_default_model_and_prepends_system():
    llm = FakeLLM()
    svc = ChatService(llm, "meta/default")
    out = await svc.chat([Message(Role.USER, "hello")], system="be brief")
    assert out.text == "hi"
    call = llm.calls[0]
    assert call["model"] == "meta/default"
    assert call["messages"][0].role is Role.SYSTEM and call["messages"][0].text == "be brief"


@pytest.mark.anyio
async def test_chat_requires_messages():
    with pytest.raises(InvalidRequestError):
        await ChatService(FakeLLM(), "m").chat([])


@pytest.mark.anyio
async def test_extract_json_parses_fenced_json():
    svc = ChatService(FakeLLM('```json\n{"items": [{"name": "cake", "qty": 2}]}\n```'), "m")
    data, _ = await svc.extract_json("two cakes", "items")
    assert data == {"items": [{"name": "cake", "qty": 2}]}


@pytest.mark.anyio
async def test_extract_json_rejects_non_json():
    with pytest.raises(LLMProviderError):
        await ChatService(FakeLLM("sorry, no"), "m").extract_json("x", "y")


@pytest.mark.anyio
async def test_list_models_filters_meta_and_sorts():
    models = await ChatService(FakeLLM(), "m").list_models()
    assert [m.id for m in models] == ["meta/a", "meta/b"]


@pytest.fixture
def anyio_backend():
    return "asyncio"
