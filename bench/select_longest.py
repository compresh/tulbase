"""Select the longest conversation from LMSYS-Chat-1M.

Two source modes:

  --source local --input-file path/to/lmsys_*.jsonl.gz
      Scan a local gzip-jsonl dump (e.g. compresh/datasets/lmsys/lmsys_100k.jsonl.gz).
      Schema expected: {id, conversation_id, turn_count, total_chars,
      language, model, messages: [{role, content}, ...]}.

  --source hf  (default if no --input-file given)
      Stream-scan HuggingFace `lmsys/lmsys-chat-1m` (gated, needs HF_TOKEN).
      Schema: {conversation_id, turn (int), language, model,
      conversation: [{role, content}, ...]}.

Both modes keep the top-N by turn count, normalize the longest one
into our internal format ({conversation_id, turn, conversation: [...]}),
and write to --output for the runner.

Usage:

    # Local (fast, no HF token):
    python select_longest.py --source local \\
        --input-file compresh/datasets/lmsys/lmsys_100k.jsonl.gz \\
        --output data/lmsys_longest.json

    # HF (1M-row scan, ~30-60 min):
    python select_longest.py --source hf --output data/lmsys_longest.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

DATASET_REPO = "lmsys/lmsys-chat-1m"


# ---------------------------------------------------------------------------
# Source iterators — normalize each row to a common shape:
#   {conversation_id, turn, language, model, conversation: [{role, content}]}
# ---------------------------------------------------------------------------


def _iter_local(path: Path) -> Iterable[dict]:
    """Yield rows from a local jsonl(.gz) dump (compresh/datasets/lmsys/...)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[arg-type]
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield {
                "conversation_id": r.get("conversation_id") or r.get("id"),
                "turn": r.get("turn_count") or r.get("turn") or 0,
                "language": r.get("language"),
                "model": r.get("model"),
                "conversation": r.get("messages") or r.get("conversation") or [],
                "total_chars": r.get("total_chars"),
            }


def _iter_hf() -> Iterable[dict]:
    """Yield rows from HuggingFace lmsys/lmsys-chat-1m (streaming)."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit("ERROR: pip install datasets")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        logger.warning(
            "HF_TOKEN not set — gated dataset access will likely fail. "
            "Set HF_TOKEN in .env."
        )
    ds = load_dataset(DATASET_REPO, split="train", streaming=True, token=token)
    for r in ds:
        yield {
            "conversation_id": r.get("conversation_id"),
            "turn": r.get("turn") or 0,
            "language": r.get("language"),
            "model": r.get("model"),
            "conversation": r.get("conversation") or [],
            "total_chars": None,
        }


# ---------------------------------------------------------------------------
# Scan + select
# ---------------------------------------------------------------------------


def scan(
    rows: Iterable[dict],
    output_path: Path,
    *,
    top_n: int = 5,
    max_scan: int | None = None,
    language_filter: str | None = "English",
    source_label: str = "local",
) -> dict:
    """Scan rows, keep top_n by turn count, write the longest."""
    top: list[dict] = []
    n_scanned = 0
    n_kept_lang = 0
    last_log = 0

    for row in rows:
        n_scanned += 1
        if max_scan is not None and n_scanned > max_scan:
            break
        if language_filter and row.get("language") != language_filter:
            continue
        n_kept_lang += 1

        turn = row.get("turn", 0)
        try:
            turn = int(turn)
        except (TypeError, ValueError):
            continue

        if len(top) < top_n:
            top.append(row)
            top.sort(key=lambda r: int(r.get("turn", 0)), reverse=True)
        elif turn > int(top[-1].get("turn", 0)):
            top[-1] = row
            top.sort(key=lambda r: int(r.get("turn", 0)), reverse=True)

        if n_scanned - last_log >= 10000:
            last_log = n_scanned
            logger.info(
                "scanned=%d  kept_lang=%d  best_turn=%d",
                n_scanned, n_kept_lang,
                int(top[0].get("turn", 0)) if top else 0,
            )

    if not top:
        sys.exit("ERROR: no rows matched (check source / HF_TOKEN / language filter)")

    longest = top[0]
    logger.info(
        "DONE. scanned=%d  kept_lang=%d  best_turn=%d  conv_id=%s",
        n_scanned,
        n_kept_lang,
        longest.get("turn", 0),
        longest.get("conversation_id", "?"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "conversation_id": longest.get("conversation_id"),
        "source_model": longest.get("model"),
        "language": longest.get("language"),
        "turn": int(longest.get("turn", 0)),
        "n_messages": len(longest.get("conversation", [])),
        "total_chars": longest.get("total_chars"),
        "conversation": longest.get("conversation"),
        "_source": source_label,
        "_scan_metadata": {
            "scanned": n_scanned,
            "kept_lang": n_kept_lang,
            "language_filter": language_filter,
        },
    }
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    logger.info("wrote %s (%.1f KB)",
                output_path, output_path.stat().st_size / 1024)

    topn_path = output_path.with_name(
        output_path.stem + ".top" + str(top_n) + ".jsonl"
    )
    with topn_path.open("w", encoding="utf-8") as f:
        for r in top:
            f.write(json.dumps({
                "conversation_id": r.get("conversation_id"),
                "turn": int(r.get("turn", 0)),
                "language": r.get("language"),
                "source_model": r.get("model"),
                "n_messages": len(r.get("conversation", [])),
            }, ensure_ascii=False) + "\n")
    logger.info("wrote top-%d index: %s", top_n, topn_path)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["local", "hf"], default=None,
                   help="Source mode (local jsonl.gz or HF streaming). "
                        "Defaults to 'local' if --input-file is given, "
                        "else 'hf'.")
    p.add_argument("--input-file", default=None,
                   help="Path to local lmsys_*.jsonl(.gz) dump")
    p.add_argument("--output", required=True,
                   help="Output JSON path for the longest conversation")
    p.add_argument("--top", type=int, default=10,
                   help="Keep top-N for reference (default 10)")
    p.add_argument("--max-scan", type=int, default=None,
                   help="Cap on rows scanned (default: full dataset)")
    p.add_argument("--language", default="English",
                   help="Filter by language (default English; '' = no filter)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Auto-pick source if not given
    source = args.source or ("local" if args.input_file else "hf")
    if source == "local":
        if not args.input_file:
            sys.exit("ERROR: --source local requires --input-file")
        path = Path(args.input_file)
        if not path.exists():
            sys.exit(f"ERROR: input file not found: {path}")
        logger.info("scanning local: %s", path)
        rows = _iter_local(path)
        label = f"local:{path.name}"
    else:
        logger.info("scanning HF (streaming): %s", DATASET_REPO)
        rows = _iter_hf()
        label = f"hf:{DATASET_REPO}"

    lang = args.language if args.language else None
    scan(
        rows,
        Path(args.output),
        top_n=args.top,
        max_scan=args.max_scan,
        language_filter=lang,
        source_label=label,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
