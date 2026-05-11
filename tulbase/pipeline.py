"""Pipeline — orchestrate one turn through Phase 2.2.

Flow:

    raw_turn_text
        │
        ▼
    modality.classify  ────► Segment[]   (code, terminal, json, …)
        │                       │
        │                       ▼
        │              for each non-dialog segment:
        │                  cold_storage.save (sha256 dedup)
        │                  compression_log.save (entry row)
        │                  build CompressedRef
        │
        ▼
    dialog_text  =  raw_turn_text minus segments
        │
        ▼
    Tier1Summarizer.run  ────► summary, opens, resolves, carry_out
        │
        ▼
    TurnBox(...)  ────► render_markdown ────► live-context block
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .cold_storage import ColdStorage
from .compression_log import CompressionEntry, CompressionLog
from .modality import Segment, classify
from .provenance import Provenance

# Q matrix ve semantic_store TUL 1.0 üst katmanları. Tulbase open-source
# çekirdek bunlar olmadan da çalışır (`enable_q_matrix=False`). Patent
# kapsamındaki Compresh kapalı dağıtımına dahildir, tulbase repo'sunda
# yer almaz.
try:
    from .q_matrix import QClassification, QMatrixClassifier
    _HAS_Q_MATRIX = True
except ImportError:
    QClassification = None  # type: ignore
    QMatrixClassifier = None  # type: ignore
    _HAS_Q_MATRIX = False

try:
    from .semantic_store import SemanticStore
    _HAS_SEMANTIC_STORE = True
except ImportError:
    SemanticStore = None  # type: ignore
    _HAS_SEMANTIC_STORE = False
from .summarizer import SummarizerResult, Tier1Summarizer
from .turn_box import CompressedRef, TurnBox, render_markdown

logger = logging.getLogger(__name__)

# Modalities we actively elide. Anything else (e.g. inline_code, url) is
# left in the dialog stream — too small to compress.
_ELIDED_MODALITIES = {"code_block", "terminal_output", "json_dump", "stack_trace"}

# Modality → reason string written to compression_log.reason
_REASON_BY_MODALITY = {
    "code_block": "code modality removal",
    "terminal_output": "verbose tool output",
    "json_dump": "redundant detail",
    "stack_trace": "verbose tool output",
}

# Modality short labels for CompressedRef.summary_short.
# Keys must match the *post-rename* modality names used in CompressionEntry
# (see _compress_segment: code_block → code).
_SHORT_LABEL = {
    "code": "code block",
    "code_block": "code block",
    "terminal_output": "terminal output",
    "json_dump": "json dump",
    "stack_trace": "stack trace",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PipelineResult:
    """Everything produced for one turn."""

    turn_box: TurnBox
    entries: list[CompressionEntry] = field(default_factory=list)
    markdown: str = ""
    size_orig: int = 0
    size_compressed: int = 0

    @property
    def saving_ratio(self) -> float:
        """0.0 = no saving, 1.0 = everything compressed."""
        if self.size_orig == 0:
            return 0.0
        return 1.0 - (self.size_compressed / self.size_orig)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Compose modality + cold + log + summarizer + turn_box for one turn.

    TUL 1.0: the pipeline also runs the dialog text through the Q matrix
    classifier (episodic × affective). The per-sentence verdicts are
    aggregated into a `q_distribution` on the resulting TurnBox so
    downstream stores can apply category-aware policies (Q3 dedup, Q2
    rich encoding, etc).
    """

    def __init__(
        self,
        log: CompressionLog,
        cold: ColdStorage,
        summarizer: Optional[Tier1Summarizer] = None,
        q_classifier: Optional[QMatrixClassifier] = None,
        semantic_store: Optional[SemanticStore] = None,
        enable_q_matrix: bool = False,
    ):
        self.log = log
        self.cold = cold
        self.summarizer = summarizer or Tier1Summarizer()
        # TUL 1.0 Q matrix layer.
        # Default: kapalı (tulbase çekirdek davranışı). Açmak için iki yol:
        #   1) Explicit ``q_classifier=QMatrixClassifier()`` ver — flag
        #      otomatik True'ya geçer
        #   2) ``enable_q_matrix=True`` aç (yine de q_matrix modülü
        #      kurulu olmalı — tulbase open-source dağıtımında yok)
        # Q matrix + semantic_store TUL 1.0 üst katmanları, Compresh
        # proprietary distribution. Tulbase çekirdek bunlar olmadan
        # tamamen çalışır.
        if q_classifier is not None:
            enable_q_matrix = True
        if enable_q_matrix:
            if not _HAS_Q_MATRIX:
                raise RuntimeError(
                    "enable_q_matrix=True but the q_matrix module is not "
                    "installed. The TUL 1.0 layer (Q matrix + semantic_store) "
                    "is part of the Compresh proprietary distribution, not "
                    "the tulbase open-source core."
                )
            self.q_classifier = q_classifier or QMatrixClassifier()
        else:
            self.q_classifier = None
        # Optional — when provided AND Q matrix is enabled, Q3 (fact)
        # sentences are routed through the semantic store for cross-turn
        # dedup. If Q matrix is off, semantic_store is silently ignored
        # (no Q3 verdicts to dedup).
        self.semantic_store = semantic_store

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        turn_text: str,
        *,
        session_id: str,
        turn_idx: int,
        speaker: str,
        prev_opens: Optional[Sequence[str]] = None,
        prev_carry_out: Optional[Sequence[str]] = None,
        carry_in: Optional[Sequence[str]] = None,
        provenance: Optional[Provenance] = None,
    ) -> PipelineResult:
        """Compress one turn end-to-end.

        Parameters
        ----------
        turn_text:
            Raw text of the turn (user or assistant message body).
        session_id:
            Stable per-conversation key (e.g. "v1:cowork:abdullah").
        turn_idx:
            Monotonic turn index within the session.
        speaker:
            "user" | "assistant" | "system".
        prev_opens / prev_carry_out:
            Hints from the previous turn — drives `resolves` and
            `carry_out` continuity.
        carry_in:
            Caller-supplied list of entry IDs from earlier turns the
            speaker is explicitly building on.
        provenance:
            TUL 1.0 source attribution. Attached to the resulting
            TurnBox and propagated to every CompressionEntry produced
            in this turn. ``None`` defers to the tulbase behaviour
            (legacy 3-value speaker only). When given, the wider
            Channel surfaces in the marker header and persists in
            compression_log.metadata["provenance"].
        """
        if not session_id:
            raise ValueError("session_id is required")
        if turn_idx < 0:
            raise ValueError("turn_idx must be >= 0")
        if speaker not in {"user", "assistant", "system"}:
            raise ValueError(f"unknown speaker: {speaker!r}")

        size_orig = len(turn_text)
        segments = classify(turn_text or "")
        elided = [s for s in segments if s.modality in _ELIDED_MODALITIES]

        # 1) Compress each elided segment (cold + log).
        entries: list[CompressionEntry] = []
        refs: list[CompressedRef] = []
        for counter, seg in enumerate(elided):
            entry = self._compress_segment(
                seg,
                session_id=session_id,
                turn_idx=turn_idx,
                counter=counter,
                provenance=provenance,
            )
            entries.append(entry)
            refs.append(
                CompressedRef(
                    id=entry.id,
                    modality=entry.modality,
                    summary_short=_SHORT_LABEL.get(entry.modality, entry.modality),
                    tokens_elided=_approx_tokens(entry.size_orig),
                    retrievable=entry.retrievable,
                )
            )

        # 2) Build dialog-only text by removing elided spans.
        dialog_text = _strip_segments(turn_text, elided)

        # 3) Q matrix classification — per-sentence episodic × affective.
        # Skipped entirely when the TUL 1.0 layer is disabled (tulbase).
        q_distribution: dict[str, int] = {}
        q3_new = 0
        q3_dup = 0
        if self.q_classifier is not None:
            q_pairs: list[tuple[str, QClassification]] = (
                self.q_classifier.classify_text_pairs(dialog_text)
            )
            raw_dist: dict[str, int] = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
            for _sent, qc in q_pairs:
                raw_dist[qc.quadrant] = raw_dist.get(qc.quadrant, 0) + 1
            q_distribution = {k: v for k, v in raw_dist.items() if v > 0}

            # 3b) Q3 dedup — route fact sentences through the semantic
            # store. Hit count surfaces on TurnBox.q3_dedup_hits.
            if (
                self.semantic_store is not None
                and q_distribution.get("Q3", 0) > 0
            ):
                for sent, qc in q_pairs:
                    if qc.quadrant != "Q3":
                        continue
                    _entry, match = self.semantic_store.find_or_save(
                        sent,
                        session_id=session_id,
                        turn_idx=turn_idx,
                    )
                    if match.matched:
                        q3_dup += 1
                    else:
                        q3_new += 1
                logger.info(
                    "Q3 dedup turn=%d session=%s new=%d dup=%d",
                    turn_idx, session_id, q3_new, q3_dup,
                )

        # 4) Summarize.
        summary: SummarizerResult = self.summarizer.run(
            dialog_text,
            prev_opens=prev_opens,
            prev_carry_out=prev_carry_out,
        )

        # 5) Build turn box.
        box = TurnBox(
            turn=turn_idx,
            speaker=speaker,  # type: ignore[arg-type]
            session_id=session_id,
            summary=summary.summary,
            carry_in=list(carry_in or []),
            carry_out=summary.carry_out,
            opens=summary.opens,
            resolves=summary.resolves,
            compressed_refs=refs,
            provenance=provenance,
            q_distribution=q_distribution,
            q3_dedup_hits=q3_dup,
        )

        markdown = render_markdown(box)
        size_compressed = len(markdown)

        logger.info(
            "pipeline turn=%d session=%s elided=%d size_orig=%d size_compressed=%d",
            turn_idx,
            session_id,
            len(elided),
            size_orig,
            size_compressed,
        )

        return PipelineResult(
            turn_box=box,
            entries=entries,
            markdown=markdown,
            size_orig=size_orig,
            size_compressed=size_compressed,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compress_segment(
        self,
        seg: Segment,
        *,
        session_id: str,
        turn_idx: int,
        counter: int,
        provenance: Optional[Provenance] = None,
    ) -> CompressionEntry:
        """Cold-store the segment text, log the entry, return the row."""
        cold_obj = self.cold.save(session_id, seg.text)

        modality = seg.modality
        # Spec uses "code" not "code_block" in the entry table for simplicity.
        modality_name = "code" if modality == "code_block" else modality

        entry_id = (
            f"compr-{session_id}-T{turn_idx}-{modality_name}-{counter:03d}"
        )
        size_orig = len(seg.text.encode("utf-8"))

        # `summary` here is short and deterministic — the actual richer
        # summary lives in TurnBox.summary. This one labels the artifact.
        short_summary = _short_segment_summary(seg)

        entry = CompressionEntry(
            id=entry_id,
            session_id=session_id,
            turn_idx=turn_idx,
            modality=modality_name,
            summary=short_summary,
            reason=_REASON_BY_MODALITY.get(modality, "modality removal"),
            hash=cold_obj.hash,
            size_orig=size_orig,
            size_compressed=len(short_summary.encode("utf-8")),
            retrievable=True,
            pii_filtered=False,
            cold_path=cold_obj.relative_path,
            metadata={"confidence": seg.confidence},
            provenance=provenance,
        )
        self.log.save(entry)
        return entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_segments(text: str, segments: list[Segment]) -> str:
    """Remove segment ranges, replacing each with a single space."""
    if not segments:
        return text
    out: list[str] = []
    cursor = 0
    for s in sorted(segments, key=lambda x: x.start):
        if s.start > cursor:
            out.append(text[cursor : s.start])
        out.append(" ")
        cursor = s.end
    if cursor < len(text):
        out.append(text[cursor:])
    # Collapse runs of whitespace introduced by stripping.
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()


def _approx_tokens(n_bytes: int) -> int:
    """Cheap byte→token approximation; matches retrieval.py constant."""
    return max(1, n_bytes // 4)


def _short_segment_summary(seg: Segment) -> str:
    """One-liner used in compression_log.summary."""
    label = _SHORT_LABEL.get(seg.modality, seg.modality)
    first_line = seg.text.strip().splitlines()[0] if seg.text.strip() else ""
    if len(first_line) > 60:
        first_line = first_line[:57] + "…"
    if first_line:
        return f"{label}: {first_line}"
    return label
