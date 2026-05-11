"""Groq Llama judge — equivalence + cosine similarity.

Reads runner.py output (one record per user turn with raw + compresh
answers) and asks a Groq-hosted Llama to judge whether the two answers
are functionally equivalent. Also computes cosine similarity over
sentence-transformer embeddings as a complementary signal.

Usage:

    python judge_groq.py \\
        --pairs results/pairs-haiku-20260510.jsonl \\
        --judge-model llama-3.3-70b-versatile \\
        --output results/judged-haiku.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


JUDGE_SYSTEM = """\
You are a strict evaluator comparing two answers to the SAME user question.
Both answers were produced by the same model. The only difference is
the context the model saw:

  - RAW   : full prior conversation history, verbatim
  - COMPRESH : compressed memory of the same conversation (a summary +
               retrieval tools)

Your job: decide if the COMPRESH answer is FUNCTIONALLY EQUIVALENT to
the RAW answer.

Rules:
  - "Equivalent" means: same factual content, same recommendation, same
    code (if any), same conclusion. Phrasing differences are OK.
  - "Not equivalent" means: COMPRESH misses a key fact, contradicts RAW,
    fabricates information not in RAW, or refuses to answer when RAW
    answered.
  - If COMPRESH abstains because the needed detail was elided AND the
    raw answer relied on that detail, treat COMPRESH as honest abstention
    (mark `abstained=true`, `equivalent=false`).
  - If both abstain or both fail, mark equivalent=true (consistent
    behaviour).

Reply with ONLY a JSON object:
  {"equivalent": true|false, "abstained": true|false, "reasoning": "<one sentence>"}
"""


def _make_groq_client():
    try:
        from groq import Groq  # type: ignore
    except ImportError:
        sys.exit("ERROR: pip install groq")
    return Groq()


def _build_user(rec: dict) -> str:
    user_msg = rec.get("user_message", "")
    raw = rec.get("raw") or {}
    compresh = rec.get("compresh") or {}
    return (
        f"USER QUESTION (turn {rec.get('turn_idx', '?')}):\n{user_msg}\n\n"
        f"RAW ANSWER:\n{raw.get('answer', '')}\n\n"
        f"COMPRESH ANSWER:\n{compresh.get('answer', '')}"
    )


def _parse_verdict(raw: str) -> dict:
    """Robust JSON parse — Groq Llama may wrap output in prose."""
    raw = raw.strip()
    try:
        v = json.loads(raw)
        return {
            "equivalent": bool(v.get("equivalent", False)),
            "abstained": bool(v.get("abstained", False)),
            "reasoning": str(v.get("reasoning", "")),
        }
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0))
            return {
                "equivalent": bool(v.get("equivalent", False)),
                "abstained": bool(v.get("abstained", False)),
                "reasoning": str(v.get("reasoning", "")),
            }
        except json.JSONDecodeError:
            pass
    return {
        "equivalent": False,
        "abstained": False,
        "reasoning": f"judge JSON parse fail: {raw[:120]}",
    }


def judge_one(client, judge_model: str, rec: dict) -> dict:
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": _build_user(rec)},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content or "{}"
    verdict = _parse_verdict(raw)
    verdict["judge_tokens_in"] = resp.usage.prompt_tokens
    verdict["judge_tokens_out"] = resp.usage.completion_tokens
    verdict["judge_model"] = judge_model
    return verdict


# ---------------------------------------------------------------------------
# Cosine similarity (optional but cheap)
# ---------------------------------------------------------------------------


_EMBED_MODEL = None


def _get_embedder():
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        logger.warning("sentence-transformers not installed — skipping cosine")
        return None
    _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def cosine_similarity(a: str, b: str) -> float | None:
    model = _get_embedder()
    if model is None or not a or not b:
        return None
    import numpy as np  # type: ignore
    embs = model.encode([a, b], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True,
                   help="Path to runner.py output (.jsonl)")
    p.add_argument("--judge-model", default="llama-3.3-70b-versatile")
    p.add_argument("--output", required=True)
    p.add_argument("--throttle-s", type=float, default=0.1)
    p.add_argument("--no-cosine", action="store_true",
                   help="Skip cosine similarity computation")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("ERROR: GROQ_API_KEY not set")

    client = _make_groq_client()
    pairs_path = Path(args.pairs)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_equiv = 0
    n_abst = 0
    n_err = 0

    with pairs_path.open(encoding="utf-8") as f, \
            out_path.open("w", encoding="utf-8") as out:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1

            try:
                verdict = judge_one(client, args.judge_model, rec)
                if verdict["equivalent"]:
                    n_equiv += 1
                if verdict["abstained"]:
                    n_abst += 1
            except Exception as e:
                verdict = {
                    "equivalent": False,
                    "abstained": False,
                    "reasoning": f"judge error: {type(e).__name__}: {str(e)[:160]}",
                    "judge_tokens_in": 0,
                    "judge_tokens_out": 0,
                    "judge_model": args.judge_model,
                }
                n_err += 1
                if "rate" in str(e).lower() or "429" in str(e):
                    time.sleep(15)

            cosine = None
            if not args.no_cosine:
                cosine = cosine_similarity(
                    (rec.get("raw") or {}).get("answer", ""),
                    (rec.get("compresh") or {}).get("answer", ""),
                )
            verdict["cosine_similarity"] = cosine

            merged = {**rec, "judge": verdict}
            out.write(json.dumps(merged, ensure_ascii=False) + "\n")
            out.flush()

            if n_total % 10 == 0:
                logger.info("[%d] equiv=%d abst=%d err=%d",
                            n_total, n_equiv, n_abst, n_err)
            time.sleep(args.throttle_s)

    logger.info("DONE. n=%d equiv=%d (%.1f%%) abst=%d err=%d",
                n_total, n_equiv,
                100.0 * n_equiv / max(1, n_total), n_abst, n_err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
