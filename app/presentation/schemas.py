"""HTTP request/response models (Pydantic). Translate to/from domain entities here."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.entities import Completion, ImagePart, Message, Role


class MessageIn(BaseModel):
    role: Role = Role.USER
    content: str = ""
    image_urls: list[str] = Field(default_factory=list, description="Public URLs or data URLs")

    def to_domain(self) -> Message:
        return Message(role=self.role, text=self.content, images=tuple(ImagePart(u) for u in self.image_urls))


class ChatRequest(BaseModel):
    messages: list[MessageIn] = Field(min_length=1)
    system: str | None = None
    model: str | None = Field(default=None, examples=["meta/muse-spark-1.3"])
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)


class VisionRequest(BaseModel):
    prompt: str = "Describe this image in detail."
    image_url: str
    model: str | None = None


class ExtractRequest(BaseModel):
    text: str
    instructions: str = Field(examples=["Extract items (name, quantity) and delivery address."])
    image_url: str | None = None
    model: str | None = None


class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float | None = None


class CompletionOut(BaseModel):
    text: str
    model: str
    usage: UsageOut

    @classmethod
    def from_domain(cls, c: Completion) -> "CompletionOut":
        u = c.usage
        return cls(
            text=c.text,
            model=c.model,
            usage=UsageOut(
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                total_tokens=u.total_tokens,
                cost=u.cost,
            ),
        )


class ExtractOut(BaseModel):
    data: dict[str, Any]
    model: str
    usage: UsageOut


class ModelOut(BaseModel):
    id: str
    name: str
    context_length: int | None = None
    prompt_price: str | None = None
    completion_price: str | None = None
