"""Aggregate judged pairs into a turn-by-turn comparison report.

Reads one or more judged-*.jsonl files, computes per-model curves over
turn buckets, and writes comparison.md with summary metrics + windowed
table.

Usage:

    python aggregate.py \\
        --judged-files results/judged-haiku.jsonl results/judged-mini.jsonl \\
        --output results/comparison.md
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

WINDOW = 10  # turn bucket size


def _bucket(turn_idx: int) -> int:
    return (turn_idx // WINDOW) * WINDOW


def _safe_mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _fmt_pct(x: float | None) -> str:
    return f"{x*100:.1f}%" if x is not None else "—"


def _fmt_num(x: float | None, digits: int = 2) -> str:
    return f"{x:.{digits}f}" if x is not None else "—"


def load_records(paths: list[Path]) -> dict[str, list[dict]]:
    """Group records by ``provider/model/compresh_mode`` label.

    The compresh_mode field is TUL 1.0 multi-conv runner output. Older
    runs (without that field) fall back to ``mode=legacy`` so the report
    still renders.
    """
    out: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mode = rec.get("compresh_mode") or "legacy"
                label = (
                    f"{rec.get('provider', '?')}/"
                    f"{rec.get('model', '?')}/{mode}"
                )
                out[label].append(rec)
    return out


def summary(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}

    equiv = [bool(r.get("judge", {}).get("equivalent")) for r in records]
    abst = [bool(r.get("judge", {}).get("abstained")) for r in records]
    cos = [r.get("judge", {}).get("cosine_similarity") for r in records]
    saving = [r.get("saving_pct") for r in records]

    raw_tokens_in = [r.get("raw", {}).get("tokens_in", 0) for r in records]
    cmp_tokens_in = [r.get("compresh", {}).get("tokens_in", 0) for r in records]

    raw_lat = [r.get("raw", {}).get("latency_ms", 0) for r in records]
    cmp_lat = [r.get("compresh", {}).get("latency_ms", 0) for r in records]

    retrieval_calls = [r.get("compresh", {}).get("retrieval_calls", 0) for r in records]
    retrieval_ok = [r.get("compresh", {}).get("retrieval_ok", 0) for r in records]

    raw_err = sum(1 for r in records if r.get("raw", {}).get("error"))
    cmp_err = sum(1 for r in records if r.get("compresh", {}).get("error"))

    return {
        "n": n,
        "equivalence_rate": sum(equiv) / n,
        "abstention_rate": sum(abst) / n,
        "cosine_mean": _safe_mean(cos),
        "saving_pct_mean": _safe_mean(saving),
        "raw_tokens_in_mean": _safe_mean([float(x) for x in raw_tokens_in]),
        "compresh_tokens_in_mean": _safe_mean([float(x) for x in cmp_tokens_in]),
        "raw_latency_ms_mean": _safe_mean([float(x) for x in raw_lat]),
        "compresh_latency_ms_mean": _safe_mean([float(x) for x in cmp_lat]),
        "retrieval_calls_mean": _safe_mean([float(x) for x in retrieval_calls]),
        "retrieval_success_rate": (
            sum(retrieval_ok) / sum(retrieval_calls)
            if sum(retrieval_calls) > 0 else None
        ),
        "raw_errors": raw_err,
        "compresh_errors": cmp_err,
    }


def per_window(records: list[dict]) -> list[dict]:
    """Bucket by turn-window and compute per-bucket stats."""
    by_bucket: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_bucket[_bucket(r.get("turn_idx", 0))].append(r)

    rows = []
    for bucket in sorted(by_bucket):
        chunk = by_bucket[bucket]
        s = summary(chunk)
        s["turn_window"] = f"{bucket}-{bucket + WINDOW - 1}"
        rows.append(s)
    return rows


# ---------------------------------------------------------------------------
# TUL 1.0 — new aggregations
# ---------------------------------------------------------------------------


def marginal_cost_curve(records: list[dict]) -> list[dict]:
    """Per-user-turn marginal cost — Δraw vs Δcompresh, and cumulative.

    The depth-aware compression claim says: as the conversation deepens,
    each new turn adds less to compresh history than to raw history
    (the ratio Δcompresh / Δraw → 0). This curve is the empirical test.
    """
    rows: list[dict] = []
    cum_raw = 0
    cum_comp = 0
    for r in sorted(records, key=lambda x: x.get("user_msg_idx", 0)):
        d_raw = int(r.get("history_chars_raw_delta", 0))
        d_comp = int(r.get("history_chars_compresh_delta", 0))
        cum_raw += d_raw
        cum_comp += d_comp
        ratio = (d_comp / d_raw) if d_raw > 0 else None
        cum_saving = (
            1.0 - cum_comp / cum_raw if cum_raw > 0 else 0.0
        )
        rows.append({
            "user_msg_idx": r.get("user_msg_idx"),
            "turn_idx": r.get("turn_idx"),
            "block_label": r.get("block_label"),
            "delta_raw": d_raw,
            "delta_compresh": d_comp,
            "marginal_ratio": ratio,
            "cum_raw": cum_raw,
            "cum_compresh": cum_comp,
            "cum_saving_pct": 100.0 * cum_saving,
        })
    return rows


def q_coverage_cumulative(records: list[dict]) -> list[dict]:
    """Per-turn Q matrix verdicts plus running totals.

    Each record's ``q_distribution_this_turn`` is added to the running
    counts. Useful for showing where the categories arrive in the
    conversation — does Q3 grow steadily (factual dialog) or in bursts
    (technical answers concentrated in some blocks)?
    """
    rows: list[dict] = []
    cum = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    cum_dup = 0
    for r in sorted(records, key=lambda x: x.get("user_msg_idx", 0)):
        dist = r.get("q_distribution_this_turn") or {}
        for q in ("Q1", "Q2", "Q3", "Q4"):
            cum[q] += int(dist.get(q, 0))
        cum_dup += int(r.get("q3_dedup_hits_this_turn", 0))
        rows.append({
            "user_msg_idx": r.get("user_msg_idx"),
            "turn_idx": r.get("turn_idx"),
            "block_label": r.get("block_label"),
            "this_turn": dict(dist),
            "cum_Q1": cum["Q1"],
            "cum_Q2": cum["Q2"],
            "cum_Q3": cum["Q3"],
            "cum_Q4": cum["Q4"],
            "cum_q3_dedup": cum_dup,
        })
    return rows


def block_transition_retention(records: list[dict]) -> list[dict]:
    """At each block transition, did compresh's equivalence rate drop?

    Looks at the 5 user turns immediately before a transition vs the 5
    immediately after. Big drop ⇒ Compresh loses recall when the topic
    shifts; small drop ⇒ source attribution + Q matrix preserved enough
    context.
    """
    by_idx = sorted(records, key=lambda r: r.get("user_msg_idx", 0))
    # Find transitions — where block_label changes between consecutive
    # user turns.
    transitions: list[tuple[int, str, str]] = []
    prev_block: str | None = None
    for i, r in enumerate(by_idx):
        block = r.get("block_label")
        if prev_block is not None and block != prev_block:
            transitions.append((i, prev_block, block))
        prev_block = block

    rows: list[dict] = []
    for tx_idx, before_block, after_block in transitions:
        before = by_idx[max(0, tx_idx - 5):tx_idx]
        after = by_idx[tx_idx:tx_idx + 5]
        rows.append({
            "transition_user_msg_idx": tx_idx,
            "before_block": before_block,
            "after_block": after_block,
            "before_equiv": _safe_mean(
                [bool(r.get("judge", {}).get("equivalent")) for r in before]
            ),
            "after_equiv": _safe_mean(
                [bool(r.get("judge", {}).get("equivalent")) for r in after]
            ),
            "before_cosine": _safe_mean(
                [r.get("judge", {}).get("cosine_similarity") for r in before]
            ),
            "after_cosine": _safe_mean(
                [r.get("judge", {}).get("cosine_similarity") for r in after]
            ),
        })
    return rows


def render_md(grouped: dict[str, list[dict]]) -> str:
    out: list[str] = []
    out.append("# Turn-by-Turn Bench — Raw vs Compresh\n")
    out.append("**Method:** every user turn in a single LMSYS conversation "
               "asked once with raw history and once with Phase 2.2 compressed "
               "history. Groq Llama judges equivalence; cosine similarity is "
               "computed over MiniLM embeddings.\n\n")

    # Top-level summary table
    out.append("## Summary\n\n")
    out.append("| Model | n | Equiv | Abstain | Cosine | Saving | Retrieval/turn | Retrieval OK |\n")
    out.append("|---|---|---|---|---|---|---|---|\n")
    for label, recs in grouped.items():
        s = summary(recs)
        out.append(
            f"| `{label}` | {s['n']} | "
            f"{_fmt_pct(s.get('equivalence_rate'))} | "
            f"{_fmt_pct(s.get('abstention_rate'))} | "
            f"{_fmt_num(s.get('cosine_mean'), 3)} | "
            f"{_fmt_num(s.get('saving_pct_mean'), 1)}% | "
            f"{_fmt_num(s.get('retrieval_calls_mean'), 2)} | "
            f"{_fmt_pct(s.get('retrieval_success_rate'))} |\n"
        )

    # Token / latency comparison
    out.append("\n## Tokens & Latency\n\n")
    out.append("| Model | raw tok in | compresh tok in | raw ms | compresh ms |\n")
    out.append("|---|---|---|---|---|\n")
    for label, recs in grouped.items():
        s = summary(recs)
        out.append(
            f"| `{label}` | "
            f"{_fmt_num(s.get('raw_tokens_in_mean'), 0)} | "
            f"{_fmt_num(s.get('compresh_tokens_in_mean'), 0)} | "
            f"{_fmt_num(s.get('raw_latency_ms_mean'), 0)} | "
            f"{_fmt_num(s.get('compresh_latency_ms_mean'), 0)} |\n"
        )

    # Per-window curves
    for label, recs in grouped.items():
        out.append(f"\n## {label} — per turn window (size={WINDOW})\n\n")
        out.append("| Turn window | n | Equiv | Cosine | Saving | Retrieval/turn |\n")
        out.append("|---|---|---|---|---|---|\n")
        for row in per_window(recs):
            out.append(
                f"| {row['turn_window']} | {row['n']} | "
                f"{_fmt_pct(row.get('equivalence_rate'))} | "
                f"{_fmt_num(row.get('cosine_mean'), 3)} | "
                f"{_fmt_num(row.get('saving_pct_mean'), 1)}% | "
                f"{_fmt_num(row.get('retrieval_calls_mean'), 2)} |\n"
            )

    # --- TUL 1.0 — Marginal token cost curve ----------------------------
    for label, recs in grouped.items():
        curve = marginal_cost_curve(recs)
        if not curve or all(r["delta_raw"] == 0 for r in curve):
            continue
        out.append(
            f"\n## {label} — marginal cost (per user turn)\n\n"
        )
        out.append(
            "The depth-aware compression claim: the ratio Δcompresh / Δraw "
            "should shrink as the conversation deepens. Cumulative saving "
            "should monotonically grow toward a tunable ceiling.\n\n"
        )
        out.append("| user# | turn | block | Δraw | Δcompresh | ratio | cum Δ_raw | cum Δ_compresh | cum saving |\n")
        out.append("|---|---|---|---|---|---|---|---|---|\n")
        # Sample every other row beyond turn 20 to keep table tractable.
        for i, row in enumerate(curve):
            if i > 20 and i % 2 == 1:
                continue
            ratio = row["marginal_ratio"]
            out.append(
                f"| {row['user_msg_idx']} | {row['turn_idx']} | "
                f"{row.get('block_label') or '—'} | "
                f"{row['delta_raw']} | {row['delta_compresh']} | "
                f"{_fmt_num(ratio, 2) if ratio is not None else '—'} | "
                f"{row['cum_raw']} | {row['cum_compresh']} | "
                f"{_fmt_num(row['cum_saving_pct'], 1)}% |\n"
            )

    # --- TUL 1.0 — Q matrix coverage cumulative -----------------------
    for label, recs in grouped.items():
        cov = q_coverage_cumulative(recs)
        if not cov or all(not r["this_turn"] for r in cov):
            continue
        out.append(f"\n## {label} — Q matrix coverage (cumulative)\n\n")
        out.append(
            "How the four quadrants accumulate. Q3 (F) dominance with "
            "high `cum_q3_dedup` means the semantic store is paying for "
            "itself; Q1 (E) bursts at block transitions suggest topic "
            "shifts are landing as events, not facts.\n\n"
        )
        out.append("| user# | turn | block | this turn | cum E | cum M | cum F | cum O | cum Q3 dup |\n")
        out.append("|---|---|---|---|---|---|---|---|---|\n")
        code_map = {"Q1": "E", "Q2": "M", "Q3": "F", "Q4": "O"}
        for i, row in enumerate(cov):
            if i > 20 and i % 2 == 1:
                continue
            this_compact = " ".join(
                f"{code_map[q]}{n}"
                for q, n in row["this_turn"].items()
                if n > 0
            ) or "—"
            out.append(
                f"| {row['user_msg_idx']} | {row['turn_idx']} | "
                f"{row.get('block_label') or '—'} | "
                f"{this_compact} | "
                f"{row['cum_Q1']} | {row['cum_Q2']} | "
                f"{row['cum_Q3']} | {row['cum_Q4']} | "
                f"{row['cum_q3_dedup']} |\n"
            )

    # --- TUL 1.0 — Block transition retention -------------------------
    for label, recs in grouped.items():
        tx = block_transition_retention(recs)
        if not tx:
            continue
        out.append(f"\n## {label} — block transition retention\n\n")
        out.append(
            "At each topic boundary (block change), how the equivalence "
            "rate and cosine similarity move between the 5 turns before "
            "vs the 5 turns after. A small drop = Compresh carries "
            "context across drift; a big drop = recall fails when topic "
            "shifts.\n\n"
        )
        out.append("| transition | before block | after block | equiv before | equiv after | Δ equiv | cosine before | cosine after |\n")
        out.append("|---|---|---|---|---|---|---|---|\n")
        for row in tx:
            be = row.get("before_equiv")
            ae = row.get("after_equiv")
            d = (ae - be) if (be is not None and ae is not None) else None
            out.append(
                f"| T{row['transition_user_msg_idx']} | "
                f"{row['before_block']} | {row['after_block']} | "
                f"{_fmt_pct(be)} | {_fmt_pct(ae)} | "
                f"{_fmt_num(d * 100, 1) + 'pp' if d is not None else '—'} | "
                f"{_fmt_num(row.get('before_cosine'), 3)} | "
                f"{_fmt_num(row.get('after_cosine'), 3)} |\n"
            )

    # Quick interpretive footer
    out.append("\n## Reading the table\n\n")
    out.append(
        "- **Equiv** — Groq Llama judge says compresh answer is functionally "
        "the same as raw. Higher is better. Should stay flat across turns "
        "(claim: depth-aware compression preserves utility).\n"
        "- **Abstain** — compresh honestly refused because the elided detail "
        "wasn't retrievable. Counts AGAINST equivalence but FOR honesty — "
        "a healthy signal in late turns where retrieval is expected.\n"
        "- **Cosine** — semantic similarity of the two answers. Less strict "
        "than equivalence; complementary signal.\n"
        "- **Saving** — `(1 − compresh_chars / raw_chars)`. Should grow with "
        "turns (depth-aware claim).\n"
        "- **Retrieval/turn** — average `fetch_compressed` calls. Zero "
        "everywhere = model never thought it needed detail; high in late "
        "turns = model is using the epistemic layer correctly.\n"
    )
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--judged-files", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    paths = [Path(p) for p in args.judged_files]
    grouped = load_records(paths)
    if not grouped:
        sys.exit("ERROR: no records loaded")

    md = render_md(grouped)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    logger.info("wrote %s (%d models, %d total records)",
                out_path, len(grouped), sum(len(v) for v in grouped.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
