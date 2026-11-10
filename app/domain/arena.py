"""Debate Arena domain: personas, stances, votes and a debate run."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Persona:
    id: int
    age: int
    job: str
    place: str
    commute: str
    values: str
    temperament: str

    def describe(self) -> str:
        return (
            f"{self.age}-year-old {self.job} from {self.place}, gets around by {self.commute}, "
            f"cares most about {self.values}, temperament: {self.temperament}"
        )

    def short(self) -> str:
        return f"{self.job}, {self.age}, {self.place}"


@dataclass
class Vote:
    persona: Persona
    stance: int | None = None
    argument: str = ""
    confidence: float = 0.0
    stance2: int | None = None  # after persuasion round
    argument2: str = ""
    error: str | None = None

    @property
    def changed(self) -> bool:
        return self.stance2 is not None and self.stance is not None and self.stance2 != self.stance


@dataclass
class DebateRun:
    id: str
    question: str
    size: int
    persuade: bool
    status: str = "framing"  # framing | voting | persuading | judging | done | error
    stances: list[str] = field(default_factory=list)
    votes: list[Vote] = field(default_factory=list)
    verdict: str = ""
    cost: float = 0.0
    tokens: int = 0
    calls: int = 0
    errors: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def counts(self, second_round: bool = False) -> list[int]:
        out = [0] * len(self.stances)
        for v in self.votes:
            s = v.stance2 if second_round and v.stance2 is not None else v.stance
            if s is not None and 0 <= s < len(out):
                out[s] += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        elapsed = (self.finished_at or time.time()) - self.started_at
        changed = sum(1 for v in self.votes if v.changed)
        return {
            "id": self.id,
            "question": self.question,
            "size": self.size,
            "status": self.status,
            "stances": self.stances,
            "counts": self.counts(),
            "counts_after": self.counts(second_round=True) if self.persuade else None,
            "changed": changed,
            "verdict": self.verdict,
            "cost": round(self.cost, 6),
            "tokens": self.tokens,
            "calls": self.calls,
            "errors": self.errors,
            "error": self.error,
            "elapsed": round(elapsed, 1),
            "calls_per_sec": round(self.calls / elapsed, 1) if elapsed > 0 else 0,
            "votes": [
                {
                    "who": v.persona.short(),
                    "s": v.stance,
                    "a": v.argument,
                    "c": v.confidence,
                    "s2": v.stance2,
                    "a2": v.argument2,
                }
                for v in self.votes
            ],
        }
