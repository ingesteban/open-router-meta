"""App factory and wiring (composition root). Run: uvicorn app.main:app --reload"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.application.chat_service import ChatService
from app.core.config import get_settings
from app.domain.exceptions import (
    DomainError,
    InvalidRequestError,
    LLMAuthError,
    LLMProviderError,
    LLMRateLimitError,
)
from app.infrastructure.openrouter_client import OpenRouterClient
from app.presentation.routes import router

STATIC_DIR = Path(__file__).parent / "presentation" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    client = OpenRouterClient(
        s.openrouter_api_key,
        s.openrouter_base_url,
        app_url=s.app_url,
        app_title=s.app_title,
        timeout=s.request_timeout_seconds,
    )
    app.state.chat_service = ChatService(client, s.default_model)
    app.state.default_model = s.default_model
    yield
    await client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="Muse Build Day Starter", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)

    @app.exception_handler(DomainError)
    async def domain_error_handler(_: Request, exc: DomainError):
        if isinstance(exc, InvalidRequestError):
            code = 422
        elif isinstance(exc, LLMAuthError):
            code = 401
        elif isinstance(exc, LLMRateLimitError):
            code = 429
        elif isinstance(exc, LLMProviderError):
            code = 502
        else:
            code = 400
        return JSONResponse(status_code=code, content={"error": type(exc).__name__, "detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
