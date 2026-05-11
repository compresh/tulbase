"""Backfill — retrospective compression of an existing conversation.

The mid-conversation activation story. A user runs their agent for 30
turns without Compresh, hits a token-wall, and flips the switch. Without
backfill, the prior 30 turns are dead weight: they keep getting sent
verbatim on every request. With backfill, those 30 turns are streamed
through the pipeline batch-style — TurnBox markdown + cold storage +
compression_log entries materialize as if Compresh had been on from
turn 0.

This module lives at the tulbase layer. It does not introduce TUL 1.0
concepts (Q matrix, source-aware speaker enum, tombstone) — those will
be added as separate layers above tulbase. Backfill itself is a
primitive: take messages in, emit TurnBoxes out, deterministically.

Design:

  - **Pass 1 (forward)** — turn-by-turn through Pipeline.run(), with an
    *accumulated* set of unresolved opens / carry_out tokens fed to each
    new turn instead of only the previous turn's. This recovers
    long-distance resolves chains that the realtime pipeline misses
    (turn 5 opens X, turn 12 resolves X — only seen if accumulator
    holds X across turns 6–11).

  - **Source attribution shim** — Anthropic-style API messages where
    `content` is a list of `tool_use` / `tool_result` blocks are
    recognized and the speaker is mapped accordingly:
        tool_use   → spoken by "assistant" but tagged as tool call
        tool_result → role="user" in the wire format, but actually
                       authored by a tool. Speaker is normalized.
    This is the *minimum* source-aware shim — full multi-channel
    speaker enum is a TUL 1.0 concern.

Usage::

    from tulbase.backfill import Backfiller

    bf = Backfiller(log=..., cold=...)
    result = bf.run_batch(messages, session_id="bench-xyz")
    # result.boxes  — list[TurnBox]
    # result.results — list[PipelineResult] (entries, sizes, markdowns)
    # result.skipped_turns — turns where pipeline failed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .cold_storage import ColdStorage
from .compression_log import CompressionLog
from .pipeline import Pipeline, PipelineResult
from .provenance import Provenance
from .summarizer import Tier1Summarizer
from .turn_box import TurnBox

logger = logging.getLogger(__name__)

# Cap on the accumulator window for unresolved opens / carry_out.
# Without a cap the summarizer prev_opens/prev_carry_out lists grow
# unbounded on long conversations.
_OPENS_WINDOW = 10
_CARRY_WINDOW = 15


@dataclass(slots=True)
class BackfillResult:
    """Outcome of one backfill run."""

    boxes: list[TurnBox] = field(default_factory=list)
    results: list[PipelineResult] = field(default_factory=list)
    skipped_turns: list[int] = field(default_factory=list)
    total_size_orig: int = 0
    total_size_compressed: int = 0

    @property
    def saving_ratio(self) -> float:
        if self.total_size_orig == 0:
            return 0.0
        return 1.0 - (self.total_size_compressed / self.total_size_orig)


# ---------------------------------------------------------------------------
# Source-attribution shim
# ---------------------------------------------------------------------------


def _normalize_message(
    msg: dict, *, turn_idx: int | None = None,
) -> tuple[str, str, Provenance]:
    """Return ``(pipeline_speaker, text_content, provenance)``.

    Anthropic and OpenAI both accept either a plain string content or a
    list of typed blocks. The shim:

      - Plain string → speaker=role, channel=role (user/assistant/system)
      - List with tool_use blocks → speaker="assistant", channel="assistant"
                                    (call trace is part of assistant's
                                    output; the result it triggers is
                                    a separate message)
      - List with tool_result blocks → pipeline_speaker="user"
                                       (wire format), channel="tool",
                                       source_name=tool name (recovered
                                       from tool_use_id when available)
      - List with text blocks → speaker=role, channel=role
      - Sentinel hints in ``msg["channel"]`` (free-form caller-supplied
        override, e.g. ``"external"`` for a web scrape, ``"scheduled"``
        for a background task output) take precedence over wire-format
        inference.

    The returned ``pipeline_speaker`` is always one of
    ``user|assistant|system`` so it can be fed straight into
    Pipeline.run(). The wider source identity lives in ``provenance``.
    """
    role = msg.get("role") or "user"
    content = msg.get("content")
    explicit_channel = msg.get("channel")  # caller-supplied override
    explicit_source = msg.get("source_name")

    # Default channel is the role for plain messages.
    channel: str = explicit_channel or role
    source_name: Optional[str] = explicit_source

    has_tool_use = False
    has_tool_result = False
    tool_use_name: Optional[str] = None
    text_parts: list[str] = []

    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    elif not isinstance(content, list):
        text = str(content)
    else:
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue
            btype = block.get("type")
            if btype == "tool_use":
                has_tool_use = True
                tname = block.get("name", "?")
                if tool_use_name is None:
                    tool_use_name = tname
                tinput = block.get("input", {})
                text_parts.append(f"[tool_use {tname}({tinput})]")
            elif btype == "tool_result":
                has_tool_result = True
                inner = block.get("content", "")
                if isinstance(inner, list):
                    inner = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in inner
                    )
                text_parts.append(str(inner))
            elif btype == "text":
                text_parts.append(block.get("text", ""))
            else:
                text_parts.append(str(block))
        text = "\n".join(p for p in text_parts if p)

    # Channel resolution. Explicit hint always wins.
    if explicit_channel is None:
        if has_tool_result and not has_tool_use:
            channel = "tool"
            if source_name is None:
                # Best-effort recovery — we may have the tool_use_id but
                # not the original name; caller can pass source_name
                # explicitly. Leave as None if unknown.
                source_name = None
        elif has_tool_use:
            channel = "assistant"
            source_name = source_name or tool_use_name
        else:
            channel = role

    # Map channel → pipeline_speaker (legacy 3-value).
    if channel in ("user", "assistant", "system"):
        pipeline_speaker = channel
    else:
        # tool / external / scheduled all ride in user-role wire blocks
        pipeline_speaker = "user"

    prov = Provenance(
        channel=channel,  # type: ignore[arg-type]
        parent_turn=turn_idx,
        source_name=source_name,
    )
    return pipeline_speaker, text, prov


# ---------------------------------------------------------------------------
# Backfiller
# ---------------------------------------------------------------------------


class Backfiller:
    """Batch-compress an existing conversation through the pipeline.

    Stateless beyond the Pipeline / CompressionLog / ColdStorage it wraps:
    a new Backfiller is cheap to construct and threads do not interact.
    """

    def __init__(
        self,
        *,
        log: CompressionLog,
        cold: ColdStorage,
        summarizer: Optional[Tier1Summarizer] = None,
    ):
        self.pipeline = Pipeline(log=log, cold=cold, summarizer=summarizer)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run_batch(
        self,
        messages: Sequence[dict],
        *,
        session_id: str,
        carry_in_initial: Optional[Sequence[str]] = None,
    ) -> BackfillResult:
        """Compress ``messages`` end-to-end, returning all artefacts.

        Parameters
        ----------
        messages:
            Wire-format messages (Anthropic / OpenAI style). Each item is
            ``{"role": str, "content": str | list[block]}``.
        session_id:
            Stable key for compression_log + cold_storage namespacing.
        carry_in_initial:
            Caller-supplied carry_in for the first turn (e.g. context
            inherited from a previous session). Defaults to empty.
        """
        if not session_id:
            raise ValueError("session_id is required")

        result = BackfillResult()
        accumulated_opens: list[str] = []
        accumulated_carry: list[str] = list(carry_in_initial or [])

        for i, raw_msg in enumerate(messages):
            try:
                speaker, text, prov = _normalize_message(raw_msg, turn_idx=i)
            except Exception as e:
                logger.warning("backfill: normalize failed at T%d: %s", i, e)
                result.skipped_turns.append(i)
                continue

            try:
                pr = self.pipeline.run(
                    text,
                    session_id=session_id,
                    turn_idx=i,
                    speaker=speaker,
                    prev_opens=accumulated_opens,
                    prev_carry_out=accumulated_carry,
                    carry_in=list(accumulated_carry),
                    provenance=prov,
                )
            except Exception as e:
                logger.warning("backfill: pipeline failed at T%d: %s", i, e)
                result.skipped_turns.append(i)
                continue

            result.boxes.append(pr.turn_box)
            result.results.append(pr)
            result.total_size_orig += pr.size_orig
            result.total_size_compressed += pr.size_compressed

            # Maintain accumulator windows.
            # 1) Drop resolved opens from the accumulator.
            for q in pr.turn_box.resolves:
                if q in accumulated_opens:
                    accumulated_opens.remove(q)
            # 2) Add this turn's opens (dedup, cap window).
            for q in pr.turn_box.opens:
                if q not in accumulated_opens:
                    accumulated_opens.append(q)
            if len(accumulated_opens) > _OPENS_WINDOW:
                accumulated_opens = accumulated_opens[-_OPENS_WINDOW:]

            # 3) Refresh carry_out window with this turn's tokens.
            for tok in pr.turn_box.carry_out:
                if tok in accumulated_carry:
                    accumulated_carry.remove(tok)  # move-to-front
                accumulated_carry.append(tok)
            if len(accumulated_carry) > _CARRY_WINDOW:
                accumulated_carry = accumulated_carry[-_CARRY_WINDOW:]

        logger.info(
            "backfill done — session=%s turns=%d skipped=%d "
            "size_orig=%d size_compressed=%d saving=%.1f%%",
            session_id,
            len(result.boxes),
            len(result.skipped_turns),
            result.total_size_orig,
            result.total_size_compressed,
            100.0 * result.saving_ratio,
        )
        return result


# ---------------------------------------------------------------------------
# Functional convenience
# ---------------------------------------------------------------------------


def backfill_messages(
    messages: Sequence[dict],
    *,
    session_id: str,
    log: CompressionLog,
    cold: ColdStorage,
    summarizer: Optional[Tier1Summarizer] = None,
    carry_in_initial: Optional[Sequence[str]] = None,
) -> BackfillResult:
    """One-shot helper: construct Backfiller, run, return result."""
    bf = Backfiller(log=log, cold=cold, summarizer=summarizer)
    return bf.run_batch(
        messages, session_id=session_id, carry_in_initial=carry_in_initial
    )
