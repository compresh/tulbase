"""Build the multi-conv bench dataset.

Concatenates 5 hand-selected LMSYS-Chat-1M conversations into a single
synthetic stream. The selection covers different content types so the
TUL 1.0 Q matrix + dedup layers all get exercised:

  A: code-heavy debugging          (68 turn, ~%66 modality)
  E: persona / pedagogical roleplay (64 turn, %0 modality)
  B: financial analysis            (56 turn, Q3+Q4 mix)
  C: father/son bonding rating     (52 turn, Q4 dominant)
  D: JS setImmediate debugging     (48 turn, ~%26 modality)

Total ≈ 288 turn after sertConcat (no drift bridge). The order is
deliberately A → E → B → C → D so that the conversation transitions
from technical to persona to financial to opinion back to technical —
four topic transition points where retrieval discipline can be measured.

Run from workspace root::

    python compresh/benchmark/turn-by-turn/build_multi_conv.py \\
        --output compresh/benchmark/turn-by-turn/data/lmsys_multi_conv.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = "compresh/datasets/lmsys/lmsys_100k.jsonl.gz"

# Order of concatenation. Each block is identified by its
# LMSYS-Chat-1M conversation_id, chosen from the top-50 longest English
# conversations (see select_longest.py top-50 dump).
SELECTIONS = [
    {
        "label": "A_code_heavy",
        "conversation_id": "d564055a8b844e5faa45c3918a6f8e81",
        "notes": "code-heavy debugging (vicuna-13b, 68 turn, ~%66 modality)",
    },
    {
        "label": "E_persona_pedagogical",
        "conversation_id": "2573a1b1aa0d40539e4a1866e2d3518a",
        "notes": "kindergarten teacher persona (vicuna-13b, 64 turn)",
    },
    {
        "label": "B_financial_analysis",
        "conversation_id": "bf88bce8feeb4c1f94e0a36f0f22770e",
        "notes": "C-REITs analysis (claude-1, 56 turn, factual+opinion)",
    },
    {
        "label": "C_opinion_rating",
        "conversation_id": "e426ebf436124fdfbad2802963a04407",
        "notes": "father/son bonding ratings (alpaca-13b, 52 turn, opinion)",
    },
    {
        "label": "D_js_debug",
        "conversation_id": "87bf460d94e74c37980b5539da966919",
        "notes": "JS setImmediate debugging (vicuna-13b, 48 turn, ~%26 mod)",
    },
]


def load_selected(src_path: Path) -> dict[str, dict]:
    """Stream the source dump once, picking only the selected conv_ids."""
    targets = {sel["conversation_id"] for sel in SELECTIONS}
    found: dict[str, dict] = {}
    with gzip.open(src_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = row.get("conversation_id")
            if cid in targets and cid not in found:
                found[cid] = row
                if len(found) == len(targets):
                    break
    missing = targets - set(found.keys())
    if missing:
        sys.exit(
            f"ERROR: missing conversation_ids in source: {sorted(missing)}\n"
            "Make sure compresh/datasets/lmsys/lmsys_100k.jsonl.gz exists."
        )
    return found


def build(src_path: Path, out_path: Path) -> dict:
    """Assemble the concatenated dataset."""
    rows_by_id = load_selected(src_path)

    blocks: list[dict] = []
    merged_messages: list[dict] = []
    turn_offset = 0
    for sel in SELECTIONS:
        row = rows_by_id[sel["conversation_id"]]
        msgs = row.get("messages", []) or []
        blocks.append({
            "label": sel["label"],
            "conversation_id": sel["conversation_id"],
            "source_model": row.get("model"),
            "language": row.get("language"),
            "turn_count": row.get("turn_count"),
            "total_chars": row.get("total_chars"),
            "notes": sel["notes"],
            "starts_at_turn": turn_offset,
            "ends_at_turn": turn_offset + len(msgs) - 1,
        })
        # Tag each message with its source block so downstream metrics
        # can attribute turns to the originating LMSYS conversation
        # without re-scanning.
        for m in msgs:
            tagged = dict(m)
            tagged["_block_label"] = sel["label"]
            tagged["_source_conv_id"] = sel["conversation_id"]
            merged_messages.append(tagged)
        turn_offset += len(msgs)

    out = {
        "conversation_id": "multi-conv-tul1-bench",
        "language": "English",
        "turn": len(merged_messages),
        "n_messages": len(merged_messages),
        "total_chars": sum(len(m.get("content", "")) for m in merged_messages),
        "conversation": merged_messages,
        "_source": "lmsys-chat-1m local 100k subset (5-block concat)",
        "_blocks": blocks,
        "_transition_turn_indices": [
            b["starts_at_turn"] for b in blocks[1:]
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    logger.info(
        "wrote %s (%d msgs, %d KB)",
        out_path, len(merged_messages), out_path.stat().st_size // 1024,
    )
    print()
    print(f"Built multi-conv dataset: {out_path}")
    print(f"  Total messages: {len(merged_messages)}")
    print(f"  Total chars:    {out['total_chars']:,}")
    print(f"  Transitions at: {out['_transition_turn_indices']}")
    print(f"\nBlocks:")
    for b in blocks:
        print(
            f"  {b['label']:25s} T{b['starts_at_turn']:3d}..T{b['ends_at_turn']:3d}  "
            f"{b['turn_count']} turn  {b.get('total_chars','?'):>6} ch  "
            f"({b['notes']})"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help="Path to lmsys_100k.jsonl.gz")
    p.add_argument("--output", required=True,
                   help="Where to write the merged JSON")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    build(Path(args.source), Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
