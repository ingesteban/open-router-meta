"""Ports: interfaces the application layer depends on.

Infrastructure provides the implementations (e.g. OpenRouterClient),
so swapping providers or faking them in tests never touches use cases.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, Sequence

from app.domain.entities import Completion, Message, ModelInfo


class LLMClient(Protocol):
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> Completion: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...

    async def list_models(self) -> list[ModelInfo]: ...
