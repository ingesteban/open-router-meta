"""Debate Arena use case: frame -> N parallel votes -> (persuasion round) -> judge."""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Any

from app.application.chat_service import _parse_json
from app.domain.arena import DebateRun, Persona, Vote
from app.domain.entities import Completion, Message, Role
from app.domain.ports import LLMClient

AGES = list(range(18, 82))
JOBS = [
    "nurse", "software engineer", "barista", "teacher", "Uber driver", "retired firefighter",
    "small-business owner", "student", "construction worker", "doctor", "artist", "lawyer",
    "chef", "delivery rider", "startup founder", "librarian", "plumber", "real-estate agent",
    "social worker", "musician", "police officer", "data scientist", "farmer", "pastor",
    "stay-at-home parent", "truck driver", "professor", "hotel housekeeper", "investor", "mechanic",
]
PLACES = [
    "the Mission", "SoMa", "the Sunset", "Oakland", "Chinatown", "Pacific Heights", "the Tenderloin",
    "Berkeley", "Daly City", "Palo Alto", "the Richmond", "Bayview", "Noe Valley", "San Jose",
    "Marin", "the Castro", "Fremont", "North Beach",
]
COMMUTES = ["car", "bike", "BART", "bus", "walking", "scooter", "remote work", "motorcycle"]
VALUES = [
    "family", "money", "the environment", "freedom", "safety", "tradition", "innovation",
    "fairness", "community", "health", "efficiency", "local business",
]
TEMPERAMENTS = [
    "skeptical", "optimistic", "pragmatic", "contrarian", "idealistic", "anxious",
    "data-driven", "sarcastic", "easygoing", "stubborn",
]

FALLBACK_STANCES = ["Yes", "No", "It depends"]


def make_personas(n: int, seed: int | None = None) -> list[Persona]:
    """Free (no LLM calls): random but diverse citizens."""
    rnd = random.Random(seed)
    return [
        Persona(
            id=i,
            age=rnd.choice(AGES),
            job=rnd.choice(JOBS),
            place=rnd.choice(PLACES),
            commute=rnd.choice(COMMUTES),
            values=rnd.choice(VALUES),
            temperament=rnd.choice(TEMPERAMENTS),
        )
        for i in range(n)
    ]


class ArenaService:
    def __init__(
        self,
        llm: LLMClient,
        model: str,
        *,
        concurrency: int = 30,
        reasoning_effort: str | None = "low",
        max_runs: int = 20,
    ) -> None:
        self._llm = llm
        self._model = model
        self._concurrency = concurrency
        self._sems: dict[str, asyncio.Semaphore] = {}  # per run, created inside the event loop
        self._extra: dict[str, Any] = {"usage": {"include": True}}
        if reasoning_effort:
            self._extra["reasoning"] = {"effort": reasoning_effort}
        self._runs: dict[str, DebateRun] = {}
        self._tasks: set[asyncio.Task] = set()
        self._max_runs = max_runs

    # ---- public API -----------------------------------------------------

    def start(self, question: str, size: int, persuade: bool = True, seed: int | None = None) -> DebateRun:
        run = DebateRun(id=uuid.uuid4().hex[:8], question=question.strip(), size=size, persuade=persuade)
        run.votes = [Vote(persona=p) for p in make_personas(size, seed)]
        self._runs[run.id] = run
        if len(self._runs) > self._max_runs:  # keep memory bounded
            self._runs.pop(next(iter(self._runs)))
        task = asyncio.create_task(self.execute(run))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return run

    def get(self, run_id: str) -> DebateRun | None:
        return self._runs.get(run_id)

    async def execute(self, run: DebateRun) -> DebateRun:
        self._sems[run.id] = asyncio.Semaphore(self._concurrency)
        try:
            run.status = "framing"
            run.stances = await self._frame(run)
            run.status = "voting"
            await asyncio.gather(*(self._vote(run, v) for v in run.votes))
            if run.persuade:
                run.status = "persuading"
                await asyncio.gather(*(self._persuade(run, v) for v in run.votes))
            run.status = "judging"
            run.verdict = await self._judge(run)
            run.status = "done"
        except Exception as exc:  # surface to UI instead of dying silently
            run.status = "error"
            run.error = str(exc)
        finally:
            run.finished_at = time.time()
            self._sems.pop(run.id, None)
        return run

    # ---- steps ----------------------------------------------------------

    async def _call(self, run: DebateRun, system: str, user: str, max_tokens: int | None = None) -> Completion:
        async with self._sems[run.id]:
            c = await self._llm.complete(
                [Message(Role.SYSTEM, system), Message(Role.USER, user)],
                model=self._model,
                temperature=0.9,
                max_tokens=max_tokens,
                extra=self._extra,
            )
        run.calls += 1
        run.tokens += c.usage.total_tokens
        run.cost += c.usage.cost or 0.0
        return c

    async def _frame(self, run: DebateRun) -> list[str]:
        try:
            c = await self._call(
                run,
                "You frame public debates. Reply with ONE JSON object only.",
                f'Question: "{run.question}"\n'
                'Give 3 or 4 distinct, mutually exclusive positions a citizen could take, '
                'each max 5 words. JSON: {"stances": ["...", "..."]}',
            )
            stances = [str(s).strip() for s in _parse_json(c.text).get("stances", []) if str(s).strip()]
            return stances[:4] if len(stances) >= 2 else FALLBACK_STANCES
        except Exception:
            return FALLBACK_STANCES

    def _options(self, run: DebateRun) -> str:
        return "\n".join(f"{i}: {s}" for i, s in enumerate(run.stances))

    async def _vote(self, run: DebateRun, v: Vote) -> None:
        try:
            c = await self._call(
                run,
                "You role-play one specific person in a city-wide debate. Stay in character, "
                "be concrete and personal. Reply with ONE JSON object only.",
                f"You are: {v.persona.describe()}.\n"
                f'Question: "{run.question}"\nPositions:\n{self._options(run)}\n'
                'Pick the position THIS person would really take. JSON: '
                '{"stance": <number>, "argument": "<max 18 words, first person, specific to your life>", '
                '"confidence": <0.0-1.0>}',
            )
            d = _parse_json(c.text)
            stance = int(d.get("stance"))
            if not 0 <= stance < len(run.stances):
                raise ValueError("stance out of range")
            v.stance, v.argument = stance, str(d.get("argument", ""))[:200]
            v.confidence = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
        except Exception as exc:
            v.error = str(exc)
            run.errors += 1

    def _best_counterargument(self, run: DebateRun, v: Vote) -> Vote | None:
        others = [o for o in run.votes if o.stance is not None and o.stance != v.stance and o.argument]
        if not others:
            return None
        top = sorted(others, key=lambda o: o.confidence, reverse=True)[:25]
        return random.choice(top)

    async def _persuade(self, run: DebateRun, v: Vote) -> None:
        if v.stance is None:
            return
        rival = self._best_counterargument(run, v)
        if rival is None:
            v.stance2 = v.stance
            return
        try:
            c = await self._call(
                run,
                "You role-play one specific person who just heard a neighbor's argument. Weigh it honestly "
                "against your own reason, as a real open-minded person would: if it is more convincing for "
                "someone like you, switch; if not, stay. Reply with ONE JSON object only.",
                f"You are: {v.persona.describe()}.\n"
                f'Question: "{run.question}"\nPositions:\n{self._options(run)}\n'
                f'You chose {v.stance} ("{run.stances[v.stance]}") because: "{v.argument}"\n'
                f'A {rival.persona.short()} who chose {rival.stance} says: "{rival.argument}"\n'
                'Having heard that, which position do you hold now? It can be any of the positions, not only '
                'yours or theirs. Final answer? JSON: {"stance": <number>, "argument": "<max 15 words: why you stayed or switched>"}',
            )
            d = _parse_json(c.text)
            s2 = int(d.get("stance"))
            v.stance2 = s2 if 0 <= s2 < len(run.stances) else v.stance
            v.argument2 = str(d.get("argument", ""))[:200]
        except Exception:
            v.stance2 = v.stance
            run.errors += 1

    async def _judge(self, run: DebateRun) -> str:
        before, after = run.counts(), run.counts(second_round=run.persuade)
        lines = []
        for i, s in enumerate(run.stances):
            best = sorted(
                (v for v in run.votes if v.stance == i and v.argument), key=lambda v: v.confidence, reverse=True
            )[:4]
            args = "; ".join(f'{v.persona.short()}: "{v.argument}"' for v in best)
            lines.append(f"- {s}: {before[i]} before, {after[i]} after. Top arguments: {args}")
        switchers = [v for v in run.votes if v.changed][:12]
        sw = "; ".join(
            f"{v.persona.short()} {run.stances[v.stance]}->{run.stances[v.stance2]}: \"{v.argument2}\""
            for v in switchers
        )
        c = await self._call(
            run,
            "You are an impartial debate judge writing for a live audience. Be crisp and vivid.",
            f'Question: "{run.question}"\n{len(run.votes)} citizens voted.\n' + "\n".join(lines)
            + (f"\nMind changes ({sum(1 for v in run.votes if v.changed)} total), examples: {sw}" if run.persuade else "")
            + "\n\nWrite the verdict in max 90 words: the winning position, the single strongest argument "
            "on each side, and one surprising pattern (who switched or who split). Plain text, no markdown.",
        )
        return c.text.strip()
