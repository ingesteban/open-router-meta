"""Debate Arena HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.application.arena_service import ArenaService

router = APIRouter(prefix="/api/arena", tags=["arena"])


def get_arena_service(request: Request) -> ArenaService:
    return request.app.state.arena_service


class StartArena(BaseModel):
    question: str = Field(min_length=3, max_length=300, examples=["Should SF ban cars downtown?"])
    size: int = Field(default=300, ge=1)
    persuade: bool = True
    seed: int | None = None


@router.post("")
async def start(req: StartArena, request: Request, svc: ArenaService = Depends(get_arena_service)):
    max_agents = getattr(request.app.state, "arena_max_agents", 2000)
    if req.size > max_agents:
        raise HTTPException(422, f"size must be <= {max_agents}")
    run = svc.start(req.question, req.size, persuade=req.persuade, seed=req.seed)
    return {"id": run.id}


@router.get("/{run_id}")
async def state(run_id: str, svc: ArenaService = Depends(get_arena_service)):
    run = svc.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run.to_dict()
