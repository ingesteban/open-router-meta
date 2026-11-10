"""API tests: real FastAPI app, ChatService swapped for a fake via dependency_overrides."""

from fastapi.testclient import TestClient

from app.application.chat_service import ChatService
from app.main import app
from app.presentation.dependencies import get_chat_service
from tests.test_chat_service import FakeLLM


def client_with(reply: str = "hi") -> TestClient:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(FakeLLM(reply), "meta/default")
    return TestClient(app)


def test_health():
    assert client_with().get("/api/health").json() == {"status": "ok"}


def test_chat():
    r = client_with("hello!").post("/api/chat", json={"messages": [{"content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["text"] == "hello!" and r.json()["model"] == "meta/default"


def test_chat_validation():
    assert client_with().post("/api/chat", json={"messages": []}).status_code == 422


def test_stream():
    r = client_with().post("/api/chat/stream", json={"messages": [{"content": "hi"}]})
    assert r.status_code == 200 and r.text == "ab"


def test_extract():
    r = client_with('{"qty": 3}').post("/api/extract", json={"text": "three", "instructions": "qty"})
    assert r.status_code == 200 and r.json()["data"] == {"qty": 3}


def test_bad_json_maps_to_502():
    r = client_with("nope").post("/api/extract", json={"text": "x", "instructions": "y"})
    assert r.status_code == 502 and r.json()["error"] == "LLMProviderError"


def test_index_page():
    r = client_with().get("/")
    assert r.status_code == 200 and "Muse Build Day Starter" in r.text
