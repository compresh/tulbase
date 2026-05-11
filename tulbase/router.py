"""tulbase chat completions proxy router.

Accepts OpenAI-compatible /v1/chat/completions requests, compresses
the conversation history via tulbase Pipeline, forwards to the
upstream provider (OpenAI / Anthropic / OpenRouter), and returns the
response unchanged.

The user's Bearer token is forwarded to the upstream as-is — tulbase
does not manage credits, billing, or user accounts. That layer lives
in the Compresh product distribution.

Stateless per-request: each request gets a fresh temp DuckDB + cold
storage. Retrieval (fetch_compressed) works within request scope.
Stateful session caching (across requests) is a future feature.
"""

import os
import shutil
import tempfile
import time
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tulbase.cold_storage import ColdStorage
from tulbase.compression_log import CompressionLog
from tulbase.pipeline import Pipeline
from tulbase.system_prompts import HONESTY_SYSTEM_PROMPT_MINI
from tulbase.turn_box import render_markdown_many

router = APIRouter()
_client: Optional[httpx.AsyncClient] = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _detect_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("claude") or "anthropic" in m:
        return "anthropic"
    if "/" in m:
        return "openrouter"
    return "openai"


def _provider_url(provider: str) -> str:
    if provider == "anthropic":
        return "https://api.anthropic.com/v1/messages"
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1/chat/completions"
    return "https://api.openai.com/v1/chat/completions"


def _build_headers(provider: str, api_key: str) -> dict:
    if provider == "anthropic":
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    return {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions with tulbase compression."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    api_key = auth[7:].strip()
    if not api_key:
        raise HTTPException(401, "Empty API key")

    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "")
    is_stream = body.get("stream", False)
    if not messages or not model:
        raise HTTPException(400, "messages and model required")

    session_id = request.headers.get(
        "x-tulbase-session", f"req-{int(time.time() * 1000)}"
    )

    workdir = tempfile.mkdtemp(prefix="tulbase-req-")
    try:
        log = CompressionLog(os.path.join(workdir, "log.duckdb"))
        log.ensure_schema()
        cold = ColdStorage(os.path.join(workdir, "cold"))
        pipeline = Pipeline(log=log, cold=cold, enable_q_matrix=False)

        turn_boxes = []
        for i, m in enumerate(messages):
            speaker = m.get("role", "user")
            if speaker not in ("user", "assistant", "system"):
                speaker = "user"
            text = m.get("content", "") or ""
            pr = pipeline.run(
                text,
                session_id=session_id,
                turn_idx=i,
                speaker=speaker,
            )
            turn_boxes.append(pr.turn_box)

        if len(turn_boxes) > 1:
            prior_boxes = turn_boxes[:-1]
            compressed_md = render_markdown_many(prior_boxes)
            last_user_idx = max(
                (i for i, m in enumerate(messages) if m.get("role") == "user"),
                default=len(messages) - 1,
            )
            last_user_msg = messages[last_user_idx]
            sys_msgs = [
                m for m in messages[:last_user_idx] if m.get("role") == "system"
            ]
            base_system = (
                sys_msgs[0]["content"] if sys_msgs else "You are a helpful assistant."
            )
            new_system = (
                f"{base_system}\n\n"
                f"Below is a compressed memory of the conversation so far:\n\n"
                f"{compressed_md}\n\n"
                f"{HONESTY_SYSTEM_PROMPT_MINI}"
            )
            optimized = [
                {"role": "system", "content": new_system},
                last_user_msg,
            ]
        else:
            optimized = messages

        provider = _detect_provider(model)
        url = _provider_url(provider)
        headers = _build_headers(provider, api_key)

        forward_body = dict(body)
        forward_body["messages"] = optimized

        client = await get_client()

        if is_stream:
            async def stream_iter():
                async with client.stream(
                    "POST", url, json=forward_body, headers=headers
                ) as resp:
                    async for chunk in resp.aiter_raw():
                        yield chunk
            return StreamingResponse(stream_iter(), media_type="text/event-stream")

        resp = await client.post(url, json=forward_body, headers=headers)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    finally:
        shutil.rmtree(workdir, ignore_errors=True)
