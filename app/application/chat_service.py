"""Use cases. Depend only on domain entities and the LLMClient port.

Add your Build Day feature here as a new method (or a new service file):
e.g. `extract_order(voice_text)`, `screenshot_to_html(image)`, etc.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Sequence

from app.domain.entities import Completion, ImagePart, Message, ModelInfo, Role
from app.domain.exceptions import InvalidRequestError, LLMProviderError
from app.domain.ports import LLMClient


class ChatService:
    def __init__(self, llm: LLMClient, default_model: str) -> None:
        self._llm = llm
        self._default_model = default_model

    def _model(self, model: str | None) -> str:
        return model or self._default_model

    @staticmethod
    def _with_system(messages: Sequence[Message], system: str | None) -> list[Message]:
        if not messages:
            raise InvalidRequestError("At least one message is required.")
        msgs = list(messages)
        if system:
            msgs.insert(0, Message(role=Role.SYSTEM, text=system))
        return msgs

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        return await self._llm.complete(
            self._with_system(messages, system),
            model=self._model(model),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        return self._llm.stream(
            self._with_system(messages, system),
            model=self._model(model),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def describe_image(
        self, prompt: str, image_url: str, *, model: str | None = None
    ) -> Completion:
        if not image_url:
            raise InvalidRequestError("image_url is required.")
        msg = Message(role=Role.USER, text=prompt, images=(ImagePart(url=image_url),))
        return await self._llm.complete([msg], model=self._model(model))

    async def extract_json(
        self,
        text: str,
        instructions: str,
        *,
        image_url: str | None = None,
        model: str | None = None,
    ) -> tuple[dict[str, Any], Completion]:
        """Turn free text (and optionally an image) into a JSON object."""
        system = (
            "You extract structured data. Reply with ONE valid JSON object and nothing else.\n"
            f"Instructions: {instructions}"
        )
        images = (ImagePart(url=image_url),) if image_url else ()
        msgs = [Message(role=Role.SYSTEM, text=system), Message(role=Role.USER, text=text, images=images)]
        completion = await self._llm.complete(msgs, model=self._model(model), json_mode=True)
        return _parse_json(completion.text), completion

    async def list_models(self, provider_prefix: str | None = "meta/") -> list[ModelInfo]:
        models = await self._llm.list_models()
        if provider_prefix:
            models = [m for m in models if m.id.startswith(provider_prefix)]
        return sorted(models, key=lambda m: m.id)


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.removeprefix("json").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise LLMProviderError("Model did not return a JSON object.")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMProviderError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMProviderError("Model returned JSON that is not an object.")
    return data
