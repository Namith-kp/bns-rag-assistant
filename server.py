"""
server.py — FastAPI backend for the BNS 2023 RAG assistant.

Exposes:
  POST /chat/stream  — SSE stream with per-stage events (body: {query, history})
  GET  /health       — model availability check
  Static files from /static/
"""

import json
import math
import time
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

# ── Pipeline imports ──────────────────────────────────────────────────────────
from rag_pipeline import (
    CHAT_MODEL,
    EMBED_MODEL,
    BASE_URL,
    API_KEY,
    TOP_K_FETCH,
    MAX_CHUNK_CHARS,
    MAX_CONTEXT_CHARS,
    MAX_TOKENS,
    TEMPERATURE,
    RETRIEVAL_THRESHOLD,
    _SYSTEM_PROMPT,
    _NOT_COVERED,
    _expand_query,
    _embed,
    _rerank,
    _composite_score,
    _is_weak_answer,
    _fallback_extract,
    _openai_client,
    _get_collection,
    ConversationState,
    is_low_content_query,
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = []


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="BNS 2023 RAG API")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


# ── Health endpoint ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{BASE_URL}/models")
            resp.raise_for_status()
            loaded = {m["id"] for m in resp.json().get("data", [])}
    except Exception as e:
        return {"status": "error", "detail": str(e), "chat_ok": False, "embed_ok": False}

    chat_ok  = CHAT_MODEL  in loaded
    embed_ok = EMBED_MODEL in loaded
    return {
        "status":        "ok" if (chat_ok and embed_ok) else "degraded",
        "chat_model":    CHAT_MODEL,
        "embed_model":   EMBED_MODEL,
        "chat_ok":       chat_ok,
        "embed_ok":      embed_ok,
        "loaded_models": sorted(loaded),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, math.ceil(len(text) / 4))


def _build_user_message_staged(query: str, excerpts: list[str]) -> str:
    parts, total = [], 0
    for i, chunk in enumerate(excerpts, 1):
        trimmed = chunk.strip()[:MAX_CHUNK_CHARS]
        if total + len(trimmed) > MAX_CONTEXT_CHARS:
            break
        parts.append(f"Passage {i}:\n{trimmed}")
        total += len(trimmed)
    passages_text = "\n\n".join(parts)
    return (
        f"Passages from BNS 2023:\n\n{passages_text}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


def _chunk_similarity(chunk: str, query: str) -> float:
    """Normalised composite score (0–1 range) used as the UI similarity bar."""
    raw = _composite_score(chunk, query)
    return min(1.0, raw / 20.0)


# ── SSE stream ────────────────────────────────────────────────────────────────

async def _rag_stream(
    query: str,
    history: list[ChatMessage],
) -> AsyncGenerator[dict, None]:
    """
    Yields SSE-compatible dicts with 'event' and 'data' keys.

    Stages emitted:
      embedding    — query expansion + embedding call
      retrieval    — chunk ids, previews, similarity scores
      threshold    — emitted if top score is below RETRIEVAL_THRESHOLD (decline)
      prompt_built — estimated token count
      generating   — token-by-token stream from LM Studio
      done         — final answer + full source list
      error        — any exception
    """

    def evt(event: str, **payload) -> dict:
        return {"event": event, "data": json.dumps(payload)}

    try:
        # ── Stage 0: Build ConversationState from request history ─────────────
        history_dicts = [{"role": msg.role, "content": msg.content}
                         for msg in (history or [])]
        state = ConversationState.from_history_list(history_dicts)

        original_query  = query
        retrieval_query = state.get_retrieval_query(query)

        # ── Stage 1: Embedding ────────────────────────────────────────────────
        t0 = time.monotonic()
        expanded = _expand_query(retrieval_query)
        yield evt("embedding", status="running", expanded_query=expanded)

        query_vec = _embed(expanded)
        embed_ms = int((time.monotonic() - t0) * 1000)
        yield evt("embedding", status="done", ms=embed_ms)

        # ── Stage 2: Vector retrieval + re-rank ───────────────────────────────
        t1 = time.monotonic()
        yield evt("retrieval", status="running")

        results = _get_collection().query(
            query_embeddings=[query_vec],
            n_results=TOP_K_FETCH,
            include=["documents"],
        )
        candidates: list[str] = results["documents"][0]
        sources = _rerank(candidates, retrieval_query)

        chunk_events = []
        for i, chunk in enumerate(sources):
            score = _chunk_similarity(chunk, retrieval_query)
            preview = chunk.strip().replace("\n", " ")[:120]
            chunk_events.append({
                "id":        f"chunk_{i}",
                "preview":   preview,
                "full_text": chunk.strip(),
                "score":     round(score, 3),
                "rank":      i + 1,
            })

        retrieval_ms = int((time.monotonic() - t1) * 1000)
        yield evt(
            "retrieval",
            status="done",
            ms=retrieval_ms,
            chunks=chunk_events,
            total_candidates=len(candidates),
        )

        # ── Similarity threshold guard ────────────────────────────────────────
        top_score = _composite_score(sources[0], retrieval_query) if sources else 0.0
        print(f"[server] top_score={top_score:.3f}  threshold={RETRIEVAL_THRESHOLD}"
              f"  query={original_query!r}")

        if top_score < RETRIEVAL_THRESHOLD:
            yield evt(
                "done",
                status="done",
                answer=_NOT_COVERED,
                total_tokens=0,
                declined=True,
                top_score=round(top_score, 3),
                sources=[],
            )
            return

        # Update conversation topic now that retrieval was confident
        if not is_low_content_query(original_query):
            state.update_topic(original_query)

        # ── Stage 3: Prompt assembly ──────────────────────────────────────────
        user_msg = _build_user_message_staged(original_query, sources)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        full_prompt    = _SYSTEM_PROMPT + "\n\n" + user_msg
        token_estimate = _estimate_tokens(full_prompt)
        yield evt(
            "prompt_built",
            status="done",
            estimated_tokens=token_estimate,
            context_chars=len(full_prompt),
        )

        # ── Stage 4: Streaming generation ─────────────────────────────────────
        yield evt("generating", status="running", token_count=0)

        accumulated = ""
        token_count  = 0

        stream = _openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                accumulated += delta
                token_count += 1
                yield evt(
                    "generating",
                    status="streaming",
                    token=delta,
                    token_count=token_count,
                )

        # Apply fallback if model gave a weak answer
        final_answer = accumulated.strip()
        if _is_weak_answer(final_answer):
            final_answer = _fallback_extract(original_query, sources)

        # ── Stage 5: Done ──────────────────────────────────────────────────────
        yield evt(
            "done",
            status="done",
            answer=final_answer,
            total_tokens=token_count,
            declined=False,
            top_score=round(top_score, 3),
            sources=[
                {
                    "id":    f"chunk_{i}",
                    "text":  s.strip(),
                    "score": round(_chunk_similarity(s, retrieval_query), 3),
                }
                for i, s in enumerate(sources)
            ],
        )

    except Exception as exc:
        yield evt("error", detail=str(exc))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest = Body(...)):
    return EventSourceResponse(
        _rag_stream(request.query, request.history)
    )
