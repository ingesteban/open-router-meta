"""HTTP routes. Thin: validate input, call a use case, map the result."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.application.chat_service import ChatService
from app.domain.exceptions import DomainError
from app.presentation.dependencies import get_chat_service
from app.presentation.schemas import (
    ChatRequest,
    CompletionOut,
    ExtractOut,
    ExtractRequest,
    ModelOut,
    UsageOut,
    VisionRequest,
)

router = APIRouter(prefix="/api", tags=["llm"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/models", response_model=list[ModelOut])
async def list_models(provider: str = "meta/", svc: ChatService = Depends(get_chat_service)):
    """Models available on OpenRouter (filtered by prefix; pass provider= for all)."""
    return [ModelOut(**asdict(m)) for m in await svc.list_models(provider or None)]


@router.post("/chat", response_model=CompletionOut)
async def chat(req: ChatRequest, svc: ChatService = Depends(get_chat_service)):
    completion = await svc.chat(
        [m.to_domain() for m in req.messages],
        system=req.system,
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    return CompletionOut.from_domain(completion)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, svc: ChatService = Depends(get_chat_service)):
    """Plain-text token stream (works with fetch() + ReadableStream or curl -N)."""
    stream = svc.chat_stream(
        [m.to_domain() for m in req.messages],
        system=req.system,
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    async def body():
        try:
            async for token in stream:
                yield token
        except DomainError as exc:  # headers already sent; surface the error inline
            yield f"\n[error] {exc}"

    return StreamingResponse(body(), media_type="text/plain; charset=utf-8")


@router.post("/vision", response_model=CompletionOut)
async def vision(req: VisionRequest, svc: ChatService = Depends(get_chat_service)):
    return CompletionOut.from_domain(await svc.describe_image(req.prompt, req.image_url, model=req.model))


@router.post("/extract", response_model=ExtractOut)
async def extract(req: ExtractRequest, svc: ChatService = Depends(get_chat_service)):
    data, c = await svc.extract_json(req.text, req.instructions, image_url=req.image_url, model=req.model)
    out = CompletionOut.from_domain(c)
    return ExtractOut(data=data, model=out.model, usage=UsageOut(**out.usage.model_dump()))
