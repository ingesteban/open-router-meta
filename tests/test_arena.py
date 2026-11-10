"""Arena tests with a scripted fake LLM (no network, no credits)."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.application.arena_service import ArenaService, make_personas
from app.domain.arena import DebateRun, Vote
from app.domain.entities import Completion, Usage


class ArenaFakeLLM:
    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, model, temperature=None, max_tokens=None, json_mode=False, extra=None):
        self.calls += 1
        user = messages[-1].text
        usage = Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120, cost=0.00001)
        if '"stances"' in user:
            text = json.dumps({"stances": ["Full ban", "Partial ban", "No ban"]})
        elif "Final answer?" in user:
            text = json.dumps({"stance": 1, "argument": "fair point, partial makes sense"})
        elif "Pick the position" in user:
            text = json.dumps({"stance": self.calls % 3, "argument": "my life, my reason", "confidence": 0.8})
        else:
            text = "Partial ban wins."
        return Completion(text=text, model=model, usage=usage)

    async def stream(self, *a, **k):
        yield ""

    async def list_models(self):
        return []


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_personas_are_deterministic_and_diverse():
    a, b = make_personas(50, seed=1), make_personas(50, seed=1)
    assert a == b and len({p.job for p in a}) > 10


@pytest.mark.anyio
async def test_full_run_with_persuasion():
    llm = ArenaFakeLLM()
    svc = ArenaService(llm, "meta/muse-ember", concurrency=5)
    run = DebateRun(id="t", question="Ban cars?", size=30, persuade=True)
    run.votes = [Vote(persona=p) for p in make_personas(30, seed=2)]
    await svc.execute(run)

    assert run.status == "done", run.error
    assert run.stances == ["Full ban", "Partial ban", "No ban"]
    assert sum(run.counts()) == 30 and run.errors == 0
    assert run.counts(second_round=True)[1] > run.counts()[1]  # everyone was persuaded to "Partial"
    assert run.calls == 1 + 30 + 30 + 1 and abs(run.cost - run.calls * 0.00001) < 1e-9
    assert run.verdict == "Partial ban wins."
    d = run.to_dict()
    assert d["changed"] > 0 and len(d["votes"]) == 30


@pytest.mark.anyio
async def test_bad_votes_are_counted_not_fatal():
    class Broken(ArenaFakeLLM):
        async def complete(self, messages, **kw):
            if "Pick the position" in messages[-1].text:
                return Completion(text="I refuse", model="m")
            return await super().complete(messages, **kw)

    svc = ArenaService(Broken(), "m")
    run = DebateRun(id="t", question="q?", size=5, persuade=False)
    run.votes = [Vote(persona=p) for p in make_personas(5)]
    await svc.execute(run)
    assert run.status == "done" and run.errors == 5


def test_api_start_and_poll():
    from app.main import app
    from app.presentation.arena_routes import get_arena_service

    with TestClient(app) as client:
        svc = ArenaService(ArenaFakeLLM(), "meta/muse-ember", concurrency=10)
        app.dependency_overrides[get_arena_service] = lambda: svc
        r = client.post("/api/arena", json={"question": "Ban cars downtown?", "size": 20})
        assert r.status_code == 200
        run_id = r.json()["id"]
        for _ in range(50):
            state = client.get(f"/api/arena/{run_id}").json()
            if state["status"] == "done":
                break
            time.sleep(0.05)
        assert state["status"] == "done" and sum(state["counts"]) == 20
        assert client.get("/api/arena/nope").status_code == 404
        assert client.get("/arena").status_code == 200
        app.dependency_overrides.clear()
