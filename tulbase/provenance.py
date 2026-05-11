"""Provenance — source attribution metadata.

The first TUL 1.0 layer above tulbase. Tulbase treats every piece of
content as if it came from one of three speakers (user/assistant/system).
That model is too coarse for modern agentic flows where a single "turn"
may carry:

  - A user's keyboard input
  - The model's tool call (`tool_use` block)
  - A tool's execution result (web fetch, file read, db query, …)
  - A scheduled task's output

If the proxy logs all of these as "user" or "assistant", later turns
will conflate them — the model will claim the user said something that
was actually fetched from the web. Tulving's source memory ayrımı tam
buraya denk gelir: content trace ile context trace ayrı kayıtlardır.

This module provides:

  - `Channel`  — the (extended) speaker enum, 6 values.
  - `TrustLevel` — how trustworthy the content is for downstream
    decisions (user_input is highest; model_generated is lowest).
  - `Provenance` — a dataclass attached to each information unit
    carrying channel + timestamp + parent turn + trust + citation.

Tulbase's narrower 3-value Speaker is preserved at the Pipeline / TurnBox
API. The wider Channel is the TUL 1.0 surface. A `to_pipeline_speaker`
helper maps Channel → Speaker for the legacy layer so the Pipeline
itself does not need an immediate refactor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

# Channel — TUL 1.0 source-aware enum.
#
#   user        — human keyboard / chat UI input
#   assistant   — the provider LLM's generated text
#   system      — system message (config, role prompt)
#   tool        — output of an internal tool execution (fetch_compressed,
#                 db query, file read, …)
#   external    — output of an external fetch (web search, HTTP API)
#   scheduled   — output of a scheduled / background task
Channel = Literal[
    "user",
    "assistant",
    "system",
    "tool",
    "external",
    "scheduled",
]

CHANNEL_VALUES: tuple[Channel, ...] = (
    "user", "assistant", "system", "tool", "external", "scheduled",
)


# TrustLevel — how seriously should retrieval / answering treat this
# content? Used in two places:
#   1. Conflict resolution: when user_input says X and external_fetched
#      says Y, the higher trust wins (or both are surfaced).
#   2. Honesty: model must not present low-trust content as user-stated
#      fact (source amnesia prevention).
TrustLevel = Literal[
    "user_input",         # human said it — highest trust as input
    "tool_validated",     # internal tool, deterministic output
    "external_fetched",   # web / external API — freshness-bounded
    "model_generated",    # LLM inferred — lowest trust
]

TRUST_VALUES: tuple[TrustLevel, ...] = (
    "user_input",
    "tool_validated",
    "external_fetched",
    "model_generated",
)


# Channel → default TrustLevel mapping. Callers can override if they
# have stronger signal (e.g. a code-signed tool_result gets
# tool_validated; an unauthenticated scrape gets external_fetched).
_DEFAULT_TRUST: dict[Channel, TrustLevel] = {
    "user": "user_input",
    "assistant": "model_generated",
    "system": "tool_validated",
    "tool": "tool_validated",
    "external": "external_fetched",
    "scheduled": "tool_validated",
}


def default_trust(channel: Channel) -> TrustLevel:
    """Return the default trust level for a channel."""
    return _DEFAULT_TRUST.get(channel, "model_generated")


def to_pipeline_speaker(channel: Channel) -> Literal["user", "assistant", "system"]:
    """Collapse a Channel to the tulbase Pipeline's 3-value Speaker.

    The Pipeline's signature does not yet accept the wider enum (kept
    that way to avoid a tulbase-level breaking change). TUL 1.0 carries
    the full Channel in the Provenance metadata; the Speaker on the
    TurnBox stays in the legacy 3 values for now.

    Mapping:
        user, system            → as-is
        assistant               → as-is
        tool, external, scheduled → "user"  (they arrive in user-role
                                              wire blocks; the source
                                              truth lives in Provenance)
    """
    if channel in ("user", "assistant", "system"):
        return channel  # type: ignore[return-value]
    return "user"


# ---------------------------------------------------------------------------
# Provenance dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Provenance:
    """Source attribution for one information unit.

    Lives in CompressionEntry.metadata["provenance"] when persisted, and
    on TurnBox metadata when in flight. Designed to round-trip cleanly
    through JSON.

    Fields
    ------
    channel:
        Where the content came from — see `Channel`.
    fetched_at:
        When this content entered the system. For user/assistant turns
        this is the message timestamp; for tool/external it is the
        execution / fetch timestamp.
    parent_turn:
        For tool/external content: the turn index that triggered the
        fetch. The model called fetch_compressed at T16; the result
        arrived at T17, but parent_turn=16. None for unsolicited content
        (e.g. scheduled output appearing on its own).
    trust_level:
        Defaults to `default_trust(channel)`; caller can tighten or
        loosen.
    citation:
        Optional reference string — URL, entry ID, task ID, file path.
        Free-form; treated as opaque by tulbase but surfaced to the
        model for source-aware answers.
    source_name:
        Optional human-readable name of the source. For tool channels
        this is typically the tool name (e.g. "fetch_compressed",
        "browser_navigate"); for external it might be a domain.
    """

    channel: Channel
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    parent_turn: Optional[int] = None
    trust_level: Optional[TrustLevel] = None
    citation: Optional[str] = None
    source_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.channel not in CHANNEL_VALUES:
            raise ValueError(
                f"unknown channel: {self.channel!r} "
                f"(expected one of {CHANNEL_VALUES})"
            )
        if self.trust_level is None:
            self.trust_level = default_trust(self.channel)
        elif self.trust_level not in TRUST_VALUES:
            raise ValueError(
                f"unknown trust_level: {self.trust_level!r}"
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (dates as ISO strings)."""
        d = asdict(self)
        d["fetched_at"] = self.fetched_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Provenance":
        # Tolerate either string or datetime for fetched_at.
        fetched = d.get("fetched_at")
        if isinstance(fetched, str):
            fetched_dt = datetime.fromisoformat(fetched)
        elif isinstance(fetched, datetime):
            fetched_dt = fetched
        else:
            fetched_dt = datetime.now(timezone.utc)
        return cls(
            channel=d["channel"],
            fetched_at=fetched_dt,
            parent_turn=d.get("parent_turn"),
            trust_level=d.get("trust_level"),
            citation=d.get("citation"),
            source_name=d.get("source_name"),
        )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_user(cls, **kw: Any) -> "Provenance":
        return cls(channel="user", **kw)

    @classmethod
    def from_assistant(cls, **kw: Any) -> "Provenance":
        return cls(channel="assistant", **kw)

    @classmethod
    def from_tool(
        cls, name: str, *, parent_turn: Optional[int] = None, **kw: Any,
    ) -> "Provenance":
        return cls(
            channel="tool",
            source_name=name,
            parent_turn=parent_turn,
            **kw,
        )

    @classmethod
    def from_external(
        cls, source: str, *, parent_turn: Optional[int] = None, **kw: Any,
    ) -> "Provenance":
        return cls(
            channel="external",
            source_name=source,
            parent_turn=parent_turn,
            **kw,
        )

    @classmethod
    def from_scheduled(
        cls, task_id: str, **kw: Any,
    ) -> "Provenance":
        return cls(
            channel="scheduled",
            source_name=task_id,
            citation=f"scheduled:{task_id}",
            **kw,
        )
