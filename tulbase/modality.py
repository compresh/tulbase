"""Modality classifier — deterministic regex segmentation.

Splits a turn's raw text into typed segments (code_block, terminal_output,
json_dump, stack_trace, url, inline_code). The remainder is treated as
free-form dialog and is *not* compressed (it goes into the turn-box summary).

This is the LLM-free Tier-1 classifier from phase22-spec.md §4. A future
Tier-2 (Phase 2.3) will add image / OCR / doc parsing.

Design notes
------------
- Patterns run greedily; overlap resolution prefers larger / more specific
  segments. A code_block that contains an inline_code wins; a stack_trace
  that contains a url wins.
- Confidence is fixed at 0.95 for regex matches. Lower confidence is
  reserved for Tier-2 ML classifiers.
- The classifier is deterministic and side-effect-free — safe to run in
  hot path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Order matters in PATTERNS only when two patterns collide and neither
# strictly contains the other — see resolve_overlaps() for the rule.

# Specificity ranking: higher number = more specific = wins on overlap.
SPECIFICITY: dict[str, int] = {
    "code_block": 100,
    "terminal_output": 90,
    "stack_trace": 80,
    "json_dump": 70,
    "url": 30,
    "inline_code": 20,
}

PATTERNS: dict[str, str] = {
    # Triple-backtick fenced code, optional language tag.
    "code_block": r"```[\s\S]*?```",
    # Shell prompt followed by command + non-prompt lines.
    # Recognised prompts: `❯ ` (Warp/iTerm), `$ ` (bash/zsh user), `# ` (root),
    # `> ` (PowerShell / continuation). Pattern stops at next prompt or a
    # capitalized free-form line (heuristic for "back to dialog").
    "terminal_output": (
        r"(?:^|(?<=\n))(?:❯ |\$ |# |> )[^\n]+\n"
        r"(?:(?!(?:❯ |\$ |# |> ))(?!\n[A-ZŞÖÇĞÜİa-zşöçğüı]).*\n?)*"
    ),
    # Multi-line JSON object of at least 500 chars.
    "json_dump": r"\{[\s\S]{500,}?\n\}",
    # Python traceback up to a blank line or end-of-text.
    "stack_trace": r"Traceback[\s\S]+?(?=\n\n|\Z)",
    # Plain URLs (http/https).
    "url": r"https?://\S+",
    # Single-backtick inline code.
    "inline_code": r"`[^`]+`",
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Segment:
    """A typed contiguous slice of a turn's text."""

    start: int
    end: int
    modality: str
    confidence: float
    text: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Segment") -> bool:
        return not (self.end <= other.start or other.end <= self.start)

    def contains(self, other: "Segment") -> bool:
        return self.start <= other.start and other.end <= self.end


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(turn_text: str) -> list[Segment]:
    """Return all detected segments, overlaps resolved, sorted by start.

    The remainder (text outside any segment) is *not* returned — callers
    treat it as `dialog` and feed it into the summarizer.
    """
    if not turn_text:
        return []

    raw: list[Segment] = []
    for modality, pat in PATTERNS.items():
        for m in re.finditer(pat, turn_text, flags=re.MULTILINE):
            raw.append(
                Segment(
                    start=m.start(),
                    end=m.end(),
                    modality=modality,
                    confidence=0.95,
                    text=m.group(0),
                )
            )
    return resolve_overlaps(raw)


def resolve_overlaps(segments: Iterable[Segment]) -> list[Segment]:
    """Drop segments that are contained in (or lose against) others.

    Rule:
      1. If A strictly contains B, B is dropped.
      2. If A and B overlap but neither contains the other, the more
         specific modality (per `SPECIFICITY`) wins; the loser is dropped.
      3. Ties broken by larger length, then earlier start.
    """
    seg_list = sorted(segments, key=lambda s: (s.start, -s.length))
    keep: list[Segment] = []

    for s in seg_list:
        drop_s = False
        replace_idx: int | None = None

        for i, k in enumerate(keep):
            if not s.overlaps(k):
                continue

            if k.contains(s) and k.modality != s.modality:
                # Strictly contained → s loses regardless of specificity
                # (a url inside a code_block is not separately compressed).
                drop_s = True
                break

            if s.contains(k) and s.modality != k.modality:
                # Container wins; mark old loser for replacement.
                replace_idx = i
                continue

            # Partial overlap → specificity decides.
            s_score = SPECIFICITY.get(s.modality, 0)
            k_score = SPECIFICITY.get(k.modality, 0)
            if s_score > k_score:
                replace_idx = i
            elif s_score < k_score:
                drop_s = True
                break
            else:
                # Tie — prefer longer; on length tie, prefer earlier start.
                if s.length > k.length:
                    replace_idx = i
                else:
                    drop_s = True
                    break

        if drop_s:
            continue
        if replace_idx is not None:
            keep[replace_idx] = s
        else:
            keep.append(s)

    keep.sort(key=lambda s: s.start)
    return keep
