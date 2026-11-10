"""Composition root for request-scoped dependencies (FastAPI Depends)."""

from fastapi import Request

from app.application.chat_service import ChatService


def get_chat_service(request: Request) -> ChatService:
    # Built once at startup in app.main.lifespan; override in tests via dependency_overrides.
    return request.app.state.chat_service
