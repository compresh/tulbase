"""TurnBox — internal JSON model + markdown renderer.

A "turn box" is the live-context unit (Layer 1) the model sees. It carries:
  - `summary`: short natural-language description of the turn
  - `carry_in` / `carry_out`: open information bridges to neighbour turns
  - `opens` / `resolves`: question/answer bridges
  - `compressed_refs`: pointers to compression_log entries elided from
    this turn (with epistemic markers so the model knows what was cut)

The dataclass is the *internal* representation. `render_markdown()` turns
it into the user-facing live-context block (also what the model receives).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal, Optional

from .provenance import Provenance

Speaker = Literal["user", "assistant", "system"]


# ---------------------------------------------------------------------------
# Compressed reference (lighter than full CompressionEntry)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompressedRef:
    """A pointer the model sees in live context.

    Carries just enough info for the model to:
      a) know that something was elided (`tokens_elided`)
      b) understand roughly what (`summary_short`, `modality`)
      c) call `fetch_compressed(id)` if it needs the original
    """

    id: str
    modality: str
    summary_short: str
    tokens_elided: int
    retrievable: bool = True


# ---------------------------------------------------------------------------
# Turn box
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TurnBox:
    """Live-context turn record.

    ``speaker`` stays the tulbase 3-value enum for backward compatibility.
    The wider TUL 1.0 source attribution lives in ``provenance`` (when
    provided) — if a tool result arrived in a user-role wire message,
    ``speaker`` is "user" but ``provenance.channel`` is "tool".
    """

    turn: int
    speaker: Speaker
    session_id: str
    summary: str
    carry_in: list[str] = field(default_factory=list)
    carry_out: list[str] = field(default_factory=list)
    opens: list[str] = field(default_factory=list)
    resolves: list[str] = field(default_factory=list)
    compressed_refs: list[CompressedRef] = field(default_factory=list)
    raw_quote_optional: Optional[str] = None
    provenance: Optional[Provenance] = None
    # TUL 1.0 Q matrix distribution — counts per quadrant for the
    # dialog text of this turn. Example: {"Q1": 2, "Q3": 1, "Q4": 1}.
    # Empty when the turn has no dialog (pure modality content) or when
    # all sentences fell into a single bucket and the count is 0.
    q_distribution: dict[str, int] = field(default_factory=dict)
    # How many Q3 (fact) sentences from this turn were matched against
    # an existing semantic_store entry instead of being stored fresh.
    # 0 when the pipeline ran without a semantic_store, or when no Q3
    # content occurred. The marker renders this in parentheses.
    q3_dedup_hits: int = 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict serializes datetime objects inside Provenance as-is;
        # normalize them to ISO strings so the dict is JSON-friendly.
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        return d

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "TurnBox":
        refs = [
            CompressedRef(**r) if not isinstance(r, CompressedRef) else r
            for r in d.get("compressed_refs", [])
        ]
        prov_data = d.get("provenance")
        prov: Optional[Provenance] = None
        if prov_data is not None:
            prov = (
                prov_data
                if isinstance(prov_data, Provenance)
                else Provenance.from_dict(prov_data)
            )
        return cls(
            turn=int(d["turn"]),
            speaker=d["speaker"],
            session_id=d["session_id"],
            summary=d.get("summary", ""),
            carry_in=list(d.get("carry_in", [])),
            carry_out=list(d.get("carry_out", [])),
            opens=list(d.get("opens", [])),
            resolves=list(d.get("resolves", [])),
            compressed_refs=refs,
            raw_quote_optional=d.get("raw_quote_optional"),
            provenance=prov,
            q_distribution=dict(d.get("q_distribution") or {}),
            q3_dedup_hits=int(d.get("q3_dedup_hits") or 0),
        )

    @classmethod
    def from_json(cls, s: str) -> "TurnBox":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Markdown renderer (model-facing format from the spec)
# ---------------------------------------------------------------------------


def _speaker_label(speaker: Speaker) -> str:
    return {"user": "User", "assistant": "Assistant", "system": "System"}.get(
        speaker, speaker.title()
    )


def _channel_label(prov: Provenance) -> str:
    """Render a Provenance into a header label fragment.

    Examples
    --------
    Provenance(channel="tool", source_name="fetch_compressed")
        → "Tool:fetch_compressed"
    Provenance(channel="external", source_name="docs.anthropic.com")
        → "External:docs.anthropic.com"
    Provenance(channel="scheduled", source_name="morning-brief")
        → "Scheduled:morning-brief"
    Provenance(channel="user")  → "User"
    """
    pretty = prov.channel.title()
    if prov.source_name:
        return f"{pretty}:{prov.source_name}"
    return pretty


def _format_compressed_line(refs: list[CompressedRef]) -> str:
    """Render the model-facing 'Compressed:' line (spec §3 marker format).

    Format: ``Compressed: <n> <human label> · <modality> · <tokens> tok
    elided · ID=<id1>,<id2> · <retrievable|blocked>.``
    """
    if not refs:
        return ""
    # Group by modality for the count-prefix.
    by_modality: dict[str, list[CompressedRef]] = {}
    for r in refs:
        by_modality.setdefault(r.modality, []).append(r)

    pieces: list[str] = []
    for modality, group in by_modality.items():
        n = len(group)
        total_tok = sum(r.tokens_elided for r in group)
        ids = ",".join(r.id for r in group)
        retrievable = (
            "retrievable" if all(r.retrievable for r in group) else "blocked"
        )
        # Prefer the human-friendly label from the first ref; pluralize.
        label = group[0].summary_short or modality
        if n > 1 and not label.endswith("s"):
            label = f"{label}s"
        pieces.append(
            f"{n} {label} · {modality} · "
            f"{total_tok} tok elided · ID={ids} · {retrievable}"
        )
    return "Compressed: " + " ; ".join(pieces) + "."


def render_markdown(box: TurnBox) -> str:
    """Render a TurnBox into the live-context markdown block.

    Format (from spec §3):

        [T123 (Assistant)]
        Summary: Phase 2.1 v3 deploy script hazırlandı — rsync + ssh + docker exec.
        Compressed: 1 code block · code · 310 tok elided · ID=compr-... · retrievable.
        Opens: "v3 testi 105 QA için yeterli mi?"
    """
    # Header: use the wider source-aware label when Provenance differs
    # from the legacy 3-value speaker (e.g. tool_result rides in a
    # user-role wire message — TurnBox.speaker = "user" but
    # provenance.channel = "tool"). Without this, the model can't tell
    # whether content came from the human or from a tool execution.
    if box.provenance is not None and box.provenance.channel not in (
        box.speaker,
    ):
        label = _channel_label(box.provenance)
    else:
        label = _speaker_label(box.speaker)
    header = f"[T{box.turn} ({label})]"
    lines: list[str] = [header]

    if box.summary:
        lines.append(f"Summary: {box.summary}")

    compressed_line = _format_compressed_line(box.compressed_refs)
    if compressed_line:
        lines.append(compressed_line)

    if box.opens:
        opens_str = " ".join(f'"{q}"' for q in box.opens)
        lines.append(f"Opens: {opens_str}")

    if box.resolves:
        resolves_str = " ".join(f'"{q}"' for q in box.resolves)
        lines.append(f"Resolves: {resolves_str}")

    if box.carry_in:
        lines.append(f"Carry-in: {', '.join(box.carry_in)}")

    if box.carry_out:
        lines.append(f"Carry-out: {', '.join(box.carry_out)}")

    # TUL 1.0 Q distribution — compact code line. Order: E M F O.
    # Q3 dedup hits (when a semantic_store is wired in) surface as a
    # trailing parenthetical so the model can see "of these N facts, K
    # were already known from earlier in the conversation".
    if box.q_distribution:
        code_map = {"Q1": "E", "Q2": "M", "Q3": "F", "Q4": "O"}
        parts = []
        for q in ("Q1", "Q2", "Q3", "Q4"):
            n = box.q_distribution.get(q, 0)
            if n > 0:
                parts.append(f"{code_map[q]}={n}")
        if parts:
            suffix = ""
            if box.q3_dedup_hits > 0:
                suffix = f" ({box.q3_dedup_hits} dup)"
            lines.append(f"Q: {' '.join(parts)}{suffix}")

    return "\n".join(lines)


def render_markdown_many(boxes: Iterable[TurnBox]) -> str:
    """Render a sequence of boxes separated by blank lines."""
    return "\n\n".join(render_markdown(b) for b in boxes)
