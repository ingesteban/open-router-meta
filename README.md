# Muse Build Day Starter (FastAPI + OpenRouter)

Blank starter for the OpenRouter × Meta Build Day. Simple clean architecture by layers.

## Layers

```
app/
├── domain/            # Entities, ports (interfaces), errors. No FastAPI, no httpx.
│   ├── entities.py    #   Message, ImagePart, Completion, Usage, ModelInfo
│   ├── ports.py       #   LLMClient protocol
│   └── exceptions.py  #   DomainError → mapped to HTTP codes in main.py
├── application/       # Use cases. Depends only on domain.
│   └── chat_service.py#   chat, chat_stream, describe_image, extract_json, list_models
├── infrastructure/    # Adapters that implement the ports.
│   └── openrouter_client.py  # OpenRouter over httpx (OpenAI-compatible API)
├── presentation/      # HTTP: Pydantic schemas, routes, DI, demo page.
│   ├── schemas.py
│   ├── routes.py
│   ├── dependencies.py
│   └── static/index.html
├── core/config.py     # Settings from .env
└── main.py            # Composition root: wires adapter → service → routes
```

Dependency rule: `presentation → application → domain ← infrastructure`.

## Setup

```bash
cd ~/Projects/open-router-meta
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your OPENROUTER_API_KEY
uvicorn app.main:app --reload
```

- Demo page: http://localhost:8000 (streaming chat + image upload)
- Swagger: http://localhost:8000/docs
- Tests: `pytest -q`

## Endpoints

| Method | Path | What |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/models?provider=meta/` | Models on OpenRouter (use `provider=` for all) |
| POST | `/api/chat` | Chat completion (text + `image_urls`) |
| POST | `/api/chat/stream` | Same, streamed as plain text |
| POST | `/api/vision` | `prompt` + `image_url` → description |
| POST | `/api/extract` | Free text (+ optional image) → JSON object |

```bash
curl -s localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"content":"Say hi in 5 words"}]}'

curl -N localhost:8000/api/chat/stream -H 'Content-Type: application/json' \
  -d '{"messages":[{"content":"Write a haiku about San Francisco"}]}'

curl -s localhost:8000/api/extract -H 'Content-Type: application/json' \
  -d '{"text":"2 chocolate cakes and 1 croissant to 1 Ferry Building","instructions":"items (name, qty) and address"}'
```

`image_urls` / `image_url` accept public URLs or data URLs (`data:image/png;base64,...`).

## Adding your Build Day feature

1. Use case: add a method to `ChatService` (or a new `application/<feature>_service.py`).
2. Schema: add request/response models in `presentation/schemas.py`.
3. Route: add an endpoint in `presentation/routes.py`.
4. Test: copy the pattern in `tests/` (`FakeLLM` means no credits spent).

New provider/tool (image gen, transcription, a DB)? Add a port in `domain/ports.py`,
an adapter in `infrastructure/`, and wire it in `main.py` `lifespan`.

## 🏛️ The $1 Debate Arena (Build Day project)

**Open http://localhost:8000/arena**

Ask any question and a city of AI citizens debates it live. Every citizen is a Muse Ember call via OpenRouter.

1. **Frame**: 1 Ember call proposes 3–4 positions.
2. **Personas**: generated locally from random traits (free, no LLM calls).
3. **Vote**: N parallel Ember calls; each persona picks a position and argues it in character.
4. **Persuasion round**: each citizen hears the strongest rival argument and may switch.
5. **Verdict**: a judge call summarizes winners, best arguments and who switched.

A live counter shows real `usage.cost` from OpenRouter against the $1 budget, plus how many opinions $1 would buy.

Code: `domain/arena.py` · `application/arena_service.py` · `presentation/arena_routes.py` · `static/arena.html`.
Tune in `.env`: `ARENA_CONCURRENCY`, `ARENA_REASONING_EFFORT`, `ARENA_MAX_AGENTS`.
