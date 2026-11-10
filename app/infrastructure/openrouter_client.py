"""OpenRouter adapter: implements the LLMClient port over HTTP (OpenAI-compatible API)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Sequence

import httpx

from app.domain.entities import Completion, Message, ModelInfo, Usage
from app.domain.exceptions import LLMAuthError, LLMProviderError, LLMRateLimitError


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        *,
        app_url: str | None = None,
        app_title: str | None = None,
        timeout: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_title:
            headers["X-Title"] = app_title
        self._api_key = api_key
        self._http = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # ---- LLMClient port -------------------------------------------------

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        self._require_key()
        payload = self._payload(messages, model, temperature, max_tokens)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = await self._http.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"Network error calling OpenRouter: {exc}") from exc
        data = self._check(resp)

        if "error" in data:  # OpenRouter can return 200 with an error body
            raise LLMProviderError(str(data["error"].get("message", data["error"])))
        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError) as exc:
            raise LLMProviderError("Unexpected response shape from OpenRouter.") from exc

        u = data.get("usage") or {}
        return Completion(
            text=text,
            model=data.get("model", model),
            usage=Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
                cost=u.get("cost"),
            ),
            raw=data,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        self._require_key()
        payload = self._payload(messages, model, temperature, max_tokens)
        payload["stream"] = True
        try:
            async with self._http.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._check(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue  # skips keep-alive comments like ": OPENROUTER PROCESSING"
                    chunk = line[6:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        event = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if "error" in event:
                        raise LLMProviderError(str(event["error"].get("message", event["error"])))
                    delta = (event.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"Network error calling OpenRouter: {exc}") from exc

    async def list_models(self) -> list[ModelInfo]:
        try:
            resp = await self._http.get("/models")
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"Network error calling OpenRouter: {exc}") from exc
        data = self._check(resp)
        out: list[ModelInfo] = []
        for m in data.get("data", []):
            pricing = m.get("pricing") or {}
            out.append(
                ModelInfo(
                    id=m.get("id", ""),
                    name=m.get("name", m.get("id", "")),
                    context_length=m.get("context_length"),
                    prompt_price=pricing.get("prompt"),
                    completion_price=pricing.get("completion"),
                )
            )
        return out

    # ---- helpers --------------------------------------------------------

    def _require_key(self) -> None:
        if not self._api_key:
            raise LLMAuthError("OPENROUTER_API_KEY is not set. Add it to your .env file.", 401)

    @staticmethod
    def _payload(
        messages: Sequence[Message], model: str, temperature: float | None, max_tokens: int | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": [_to_wire(m) for m in messages]}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    @staticmethod
    def _check(resp: httpx.Response) -> dict[str, Any]:
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if resp.status_code < 400:
            return body
        err = body.get("error") if isinstance(body, dict) else None
        msg = (err.get("message") if isinstance(err, dict) else None) or resp.text or "OpenRouter error"
        if resp.status_code == 401:
            raise LLMAuthError(msg, 401)
        if resp.status_code in (402, 429):
            raise LLMRateLimitError(msg, resp.status_code)
        raise LLMProviderError(msg, resp.status_code)


def _to_wire(m: Message) -> dict[str, Any]:
    if not m.images:
        return {"role": m.role.value, "content": m.text}
    parts: list[dict[str, Any]] = []
    if m.text:
        parts.append({"type": "text", "text": m.text})
    parts += [{"type": "image_url", "image_url": {"url": img.url}} for img in m.images]
    return {"role": m.role.value, "content": parts}
