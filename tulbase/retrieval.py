"""Retrieval — fetch_compressed tool implementation.

When the model emits a `fetch_compressed(id=...)` tool call, the proxy:

  1. Looks up the entry in `compression_log` (Layer 2 index).
  2. Rejects if `retrievable == false` or `pii_filtered == true`.
  3. Reads the original bytes from cold storage (Layer 3).
  4. Optionally truncates to `max_tokens` (rough char approximation).
  5. Returns the content as a tool-call response that the proxy will
     inject into the next provider continuation.

This module is provider-agnostic. The proxy layer is responsible for
intercepting the tool call, calling `Retriever.fetch()`, and re-issuing
the request to the upstream provider with the result appended.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .cold_storage import ColdStorage
from .compression_log import CompressionEntry, CompressionLog

logger = logging.getLogger(__name__)

# Rough token approximation. We don't import tiktoken here — `counter.py`
# in production already does that and the proxy can convert before calling.
# 1 token ≈ 4 chars is the standard OpenAI rule of thumb for English/code.
DEFAULT_CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 2000


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RetrievalResult:
    """Outcome of a fetch_compressed call.

    `ok` is False when:
      - id not found
      - entry is not retrievable (pii_filtered or retrievable=false)
      - cold storage read failed

    Callers should surface `error` as the tool-call response so the model
    can react ("I have a record but cannot retrieve…") instead of
    fabricating.
    """

    ok: bool
    id: str
    content: Optional[str] = None
    modality: Optional[str] = None
    truncated: bool = False
    bytes_read: int = 0
    error: Optional[str] = None

    def to_tool_response(self) -> dict:
        """Shape suitable for OpenAI tool-call response messages."""
        if not self.ok:
            return {
                "id": self.id,
                "ok": False,
                "error": self.error or "unknown error",
            }
        return {
            "id": self.id,
            "ok": True,
            "modality": self.modality,
            "truncated": self.truncated,
            "content": self.content or "",
        }


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """Glue between compression_log and cold_storage."""

    def __init__(
        self,
        log: CompressionLog,
        cold: ColdStorage,
        *,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
    ):
        self.log = log
        self.cold = cold
        self.chars_per_token = max(1, chars_per_token)

    # ------------------------------------------------------------------
    # Tool entry point
    # ------------------------------------------------------------------
    def fetch(
        self,
        entry_id: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> RetrievalResult:
        """Lookup → permission check → cold read → optional truncation."""
        if not entry_id:
            return RetrievalResult(ok=False, id="", error="empty id")

        entry: Optional[CompressionEntry] = self.log.get(entry_id)
        if entry is None:
            logger.info("retrieval miss id=%s", entry_id)
            return RetrievalResult(ok=False, id=entry_id, error="not found")

        # PII / retrievability check.
        if entry.pii_filtered or not entry.retrievable:
            logger.info(
                "retrieval blocked id=%s pii=%s retrievable=%s",
                entry_id,
                entry.pii_filtered,
                entry.retrievable,
            )
            return RetrievalResult(
                ok=False,
                id=entry_id,
                error="entry is not retrievable (pii_filtered or blocked)",
            )

        try:
            content = self.cold.read_text(entry.session_id, entry.hash)
        except FileNotFoundError as e:
            logger.warning("retrieval cold miss id=%s err=%s", entry_id, e)
            return RetrievalResult(
                ok=False, id=entry_id, error=f"cold storage miss: {e}"
            )
        except UnicodeDecodeError as e:
            logger.warning("retrieval decode err id=%s err=%s", entry_id, e)
            return RetrievalResult(
                ok=False, id=entry_id, error=f"decode error: {e}"
            )

        bytes_read = len(content.encode("utf-8"))
        truncated = False
        char_budget = max_tokens * self.chars_per_token
        if char_budget > 0 and len(content) > char_budget:
            content = content[:char_budget] + "\n…[truncated]"
            truncated = True

        logger.info(
            "retrieval ok id=%s modality=%s bytes=%d truncated=%s",
            entry_id,
            entry.modality,
            bytes_read,
            truncated,
        )
        return RetrievalResult(
            ok=True,
            id=entry_id,
            content=content,
            modality=entry.modality,
            truncated=truncated,
            bytes_read=bytes_read,
        )

    # ------------------------------------------------------------------
    # Listing helper (for `list_compressed` tool)
    # ------------------------------------------------------------------
    def list_session(
        self,
        session_id: str,
        *,
        turn_min: Optional[int] = None,
        turn_max: Optional[int] = None,
        modality: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return a JSON-friendly listing for the model to scan."""
        rows = self.log.list_by_session(
            session_id,
            turn_min=turn_min,
            turn_max=turn_max,
            modality=modality,
            limit=limit,
        )
        return [
            {
                "id": e.id,
                "turn": e.turn_idx,
                "modality": e.modality,
                "summary": e.summary,
                "size_orig": e.size_orig,
                "retrievable": e.retrievable,
            }
            for e in rows
        ]
