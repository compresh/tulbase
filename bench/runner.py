"""Turn-by-turn dual-context runner.

For each user message in the LMSYS conversation, ask the model TWICE
with the same prompt:

  1. RAW   — full prior history as messages
  2. COMPRESH — same history compressed via Phase 2.2 (TurnBox markdown
               in system prompt + fetch_compressed/list_compressed tools)

Both answers are written to JSONL for downstream judging.

Usage:

    python runner.py \\
        --conversation data/lmsys_longest.json \\
        --provider anthropic \\
        --model claude-haiku-4-5-20251001 \\
        --output results/pairs-haiku-20260510.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# tulbase pip install ile import path otomatik halledilir

from tulbase import (  # type: ignore  # noqa: E402
    ColdStorage,
    CompressionLog,
    Pipeline,
    QMatrixClassifier,
    Retriever,
    SemanticStore,
    HONESTY_SYSTEM_PROMPT,
    HONESTY_SYSTEM_PROMPT_MINI,
    all_tools,
    all_tools_anthropic,
)
from tulbase.turn_box import render_markdown_many  # type: ignore  # noqa: E402

BASE_SYSTEM = "You are a helpful assistant."

# Maximum tool-use rounds before forcing a final answer.
MAX_TOOL_ROUNDS = 5


# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------


def _make_openai():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        sys.exit("ERROR: pip install openai")
    return OpenAI()


def _make_anthropic():
    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError:
        sys.exit("ERROR: pip install anthropic")
    return Anthropic()


def _make_openrouter():
    """OpenAI-compatible client wired to OpenRouter (200+ models, one key).

    Ranking headers (HTTP-Referer, X-Title) are optional but surface
    Compresh on OpenRouter's usage leaderboard — small marketing perk.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        sys.exit("ERROR: pip install openai")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://compre.sh",
            "X-Title": "Compresh TUL 1.0 bench",
        },
    )


# ---------------------------------------------------------------------------
# OpenAI: raw + compresh
# ---------------------------------------------------------------------------


def ask_openai_raw(client, model: str, messages: list[dict],
                   max_tokens: int = 1024) -> dict:
    """Plain inference — no tools."""
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": BASE_SYSTEM}, *messages],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return {
        "answer": resp.choices[0].message.content or "",
        "tokens_in": resp.usage.prompt_tokens,
        "tokens_out": resp.usage.completion_tokens,
        "latency_ms": int((time.monotonic() - t0) * 1000),
        "retrieval_calls": 0,
        "retrieval_ok": 0,
    }


def ask_openai_compresh(client, model: str, system: str,
                        user_message: str, retriever: Retriever,
                        session_id: str,
                        max_tokens: int = 1024) -> dict:
    """Compresh inference with tool-use loop."""
    t0 = time.monotonic()
    tools = all_tools(include_list=True)
    messages: list[dict] = [{"role": "user", "content": user_message}]
    retrieval_calls = 0
    retrieval_ok = 0
    tokens_in = 0
    tokens_out = 0
    final = ""

    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            tools=tools,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        tokens_in += resp.usage.prompt_tokens
        tokens_out += resp.usage.completion_tokens
        choice = resp.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            final = msg.content or ""
            break

        # Append the assistant message with tool_calls
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            retrieval_calls += 1
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_result = _exec_tool(retriever, tc.function.name, args, session_id)
            if tool_result.get("ok"):
                retrieval_ok += 1
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

    return {
        "answer": final,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": int((time.monotonic() - t0) * 1000),
        "retrieval_calls": retrieval_calls,
        "retrieval_ok": retrieval_ok,
    }


# ---------------------------------------------------------------------------
# Anthropic: raw + compresh
# ---------------------------------------------------------------------------


def ask_anthropic_raw(client, model: str, messages: list[dict],
                      max_tokens: int = 1024) -> dict:
    t0 = time.monotonic()
    resp = client.messages.create(
        model=model,
        system=BASE_SYSTEM,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    )
    return {
        "answer": text,
        "tokens_in": resp.usage.input_tokens,
        "tokens_out": resp.usage.output_tokens,
        "latency_ms": int((time.monotonic() - t0) * 1000),
        "retrieval_calls": 0,
        "retrieval_ok": 0,
    }


def ask_anthropic_compresh(client, model: str, system: str,
                           user_message: str, retriever: Retriever,
                           session_id: str,
                           max_tokens: int = 1024) -> dict:
    t0 = time.monotonic()
    tools = all_tools_anthropic(include_list=True)
    messages: list[dict] = [{"role": "user", "content": user_message}]
    retrieval_calls = 0
    retrieval_ok = 0
    tokens_in = 0
    tokens_out = 0
    final = ""

    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.messages.create(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        tokens_in += resp.usage.input_tokens
        tokens_out += resp.usage.output_tokens

        if resp.stop_reason != "tool_use":
            final = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )
            break

        # Append assistant turn (with tool_use blocks) verbatim
        messages.append({"role": "assistant", "content": resp.content})

        # Find tool_use blocks, execute, append tool_result blocks
        tool_results: list[dict] = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            retrieval_calls += 1
            tool_result = _exec_tool(retriever, block.name, dict(block.input), session_id)
            if tool_result.get("ok"):
                retrieval_ok += 1
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            })
        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": final,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": int((time.monotonic() - t0) * 1000),
        "retrieval_calls": retrieval_calls,
        "retrieval_ok": retrieval_ok,
    }


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def _exec_tool(retriever: Retriever, name: str, args: dict,
               session_id: str) -> dict:
    """Execute fetch_compressed / list_compressed and return JSON-friendly dict."""
    try:
        if name == "fetch_compressed":
            entry_id = args.get("id", "")
            max_tokens = int(args.get("max_tokens", 2000))
            result = retriever.fetch(entry_id, max_tokens=max_tokens)
            return {
                "ok": result.ok,
                "id": entry_id,
                "content": result.content,
                "truncated": result.truncated,
                "modality": result.modality,
                "error": result.error if not result.ok else None,
            }
        if name == "list_compressed":
            entries = retriever.list_session(
                session_id,
                turn_min=args.get("turn_min"),
                turn_max=args.get("turn_max"),
                modality=args.get("modality"),
                limit=int(args.get("limit", 100)),
            )
            return {"ok": True, "entries": entries}
        return {"ok": False, "error": f"unknown tool: {name}"}
    except Exception as e:
        return {"ok": False, "error": f"tool error: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(
    conversation: dict,
    provider: str,
    model: str,
    output_path: Path,
    *,
    max_user_turns: int | None = None,
    throttle_s: float = 0.2,
    compresh_mode: str = "tulbase",
    simulate_context_window: int | None = None,
) -> None:
    """Iterate the conversation, compress each turn, query at every user turn.

    Parameters
    ----------
    compresh_mode:
        ``"tulbase"`` — format-cleaned Phase 2.2 baseline. No Q matrix,
        no semantic dedup. Marker shows summary + compressed refs only.
        ``"tul1"`` — full TUL 1.0: Q matrix classification + semantic
        store dedup. Marker shows Q distribution + dedup hits.
    simulate_context_window:
        If set, run a controlled context-budget experiment (TUL 1.0
        Fas 1c). Before each raw-mode call, drop the oldest prior_raw
        messages until the remaining text fits in ``N`` tokens
        (approximated as ``N * 4`` characters). Compresh-mode history
        is unaffected — the whole point is to compare what happens
        when raw has to discard context while compresh holds on.
    """
    session_id = f"bench-{conversation.get('conversation_id', 'unknown')}-{compresh_mode}"
    messages = conversation.get("conversation") or []
    if not messages:
        sys.exit("ERROR: empty conversation")

    # Set up per-run in-memory state.
    workdir = Path(tempfile.mkdtemp(prefix=f"turn-by-turn-{compresh_mode}-"))
    db_path = workdir / "compression_log.duckdb"
    cold_root = workdir / "cold"
    log = CompressionLog(str(db_path))
    log.ensure_schema()
    cold = ColdStorage(str(cold_root))

    # Mode-specific pipeline wiring.
    if compresh_mode == "tul1":
        try:
            from tulbase import QMatrixClassifier, SemanticStore
        except ImportError:
            sys.exit(
                "ERROR: --compresh-mode tul1 requires the Compresh "
                "proprietary distribution (Q matrix + semantic_store). "
                "Use --compresh-mode tulbase."
            )
        sem = SemanticStore(log._conn)
        sem.ensure_schema()
        pipeline = Pipeline(
            log=log, cold=cold,
            q_classifier=QMatrixClassifier(),
            semantic_store=sem,
        )
    elif compresh_mode == "tulbase":
        pipeline = Pipeline(log=log, cold=cold, enable_q_matrix=False)
    else:
        sys.exit(f"ERROR: unknown compresh_mode {compresh_mode!r}")

    retriever = Retriever(log=log, cold=cold)

    if provider == "openai":
        client = _make_openai()
        ask_raw = ask_openai_raw
        ask_compresh = ask_openai_compresh
    elif provider == "anthropic":
        client = _make_anthropic()
        ask_raw = ask_anthropic_raw
        ask_compresh = ask_anthropic_compresh
    elif provider == "openrouter":
        # OpenAI-compatible — same code paths as OpenAI, different base URL.
        client = _make_openrouter()
        ask_raw = ask_openai_raw
        ask_compresh = ask_openai_compresh
    else:
        sys.exit(f"ERROR: unknown provider {provider!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    user_turns_done = 0
    turn_boxes: list = []
    # Cumulative deltas across user turns.
    prev_history_chars_raw = 0
    prev_history_chars_compresh = 0

    with output_path.open("w", encoding="utf-8") as out:
        for turn_idx, m in enumerate(messages):
            role = m.get("role", "")
            content = m.get("content", "") or ""
            block_label = m.get("_block_label")

            # Compress this message so subsequent turns see it as a TurnBox.
            try:
                pr = pipeline.run(
                    content,
                    session_id=session_id,
                    turn_idx=turn_idx,
                    speaker=role if role in ("user", "assistant", "system") else "user",
                )
                turn_boxes.append(pr.turn_box)
            except Exception as e:
                logger.warning("[T%d] pipeline failed: %s", turn_idx, e)
                continue

            # Only run the dual-context query at user turns with prior history.
            if role != "user" or turn_idx == 0:
                continue
            if max_user_turns is not None and user_turns_done >= max_user_turns:
                break
            user_turns_done += 1

            # Per-turn Q matrix verdict (this user message only).
            q_distribution_this_turn = dict(pr.turn_box.q_distribution)
            q3_dedup_hits_this_turn = pr.turn_box.q3_dedup_hits

            # Build the raw history for the provider. The multi-conv
            # dataset attaches `_block_label` / `_source_conv_id` to each
            # message for downstream attribution — providers reject
            # those as "Extra inputs", so strip them down to the
            # canonical role/content pair before sending.
            prior_raw_full = messages[:turn_idx]
            prior_raw = [
                {"role": m.get("role", "user"),
                 "content": m.get("content", "") or ""}
                for m in prior_raw_full
            ]
            raw_truncated_count = 0
            if simulate_context_window is not None:
                # Approximation: 1 token ≈ 4 characters.
                char_budget = simulate_context_window * 4
                while prior_raw:
                    used = sum(len(m["content"]) for m in prior_raw)
                    used += len(content)  # current user message also counts
                    if used <= char_budget:
                        break
                    prior_raw.pop(0)  # drop oldest pair element
                    raw_truncated_count += 1
            prior_boxes = turn_boxes[:turn_idx]        # compresh history before this user msg
            compresh_md = render_markdown_many(prior_boxes)
            # TUL 1.0: mini honesty fragment by default (~180 char). The
            # marker format is self-documenting and tools carry their own
            # descriptions, so the full 1.1KB onboarding text is wasted
            # on every turn. Use HONESTY_SYSTEM_PROMPT for first-turn /
            # one-shot scenarios where the model has never seen markers.
            compresh_system = (
                BASE_SYSTEM
                + "\n\nBelow is a compressed memory of the conversation so far:\n\n"
                + compresh_md
                + "\n\n"
                + HONESTY_SYSTEM_PROMPT_MINI
            )

            # Sizes (for saving %).
            raw_history_chars = sum(
                len((mm.get("content") or "")) for mm in prior_raw
            )
            compresh_history_chars = len(compresh_md)

            # Ask raw.
            try:
                raw_result = ask_raw(
                    client, model,
                    [*prior_raw, {"role": "user", "content": content}],
                )
                raw_err = None
            except Exception as e:
                raw_result = {"answer": "", "tokens_in": 0, "tokens_out": 0,
                              "latency_ms": 0, "retrieval_calls": 0,
                              "retrieval_ok": 0}
                raw_err = f"{type(e).__name__}: {str(e)[:200]}"
                logger.warning("[T%d] raw error: %s", turn_idx, raw_err)

            time.sleep(throttle_s)

            # Ask compresh.
            try:
                compresh_result = ask_compresh(
                    client, model,
                    compresh_system, content, retriever,
                    session_id,
                )
                compresh_err = None
            except Exception as e:
                compresh_result = {"answer": "", "tokens_in": 0, "tokens_out": 0,
                                   "latency_ms": 0, "retrieval_calls": 0,
                                   "retrieval_ok": 0}
                compresh_err = f"{type(e).__name__}: {str(e)[:200]}"
                logger.warning("[T%d] compresh error: %s", turn_idx, compresh_err)

            saving_pct = (
                100.0 * (1 - compresh_history_chars / raw_history_chars)
                if raw_history_chars > 0 else 0.0
            )

            # Marginal deltas across user turns — the "depth-aware
            # compression" claim hinges on compresh's delta growing
            # slower than raw's as the conversation deepens.
            history_chars_raw_delta = (
                raw_history_chars - prev_history_chars_raw
            )
            history_chars_compresh_delta = (
                compresh_history_chars - prev_history_chars_compresh
            )
            prev_history_chars_raw = raw_history_chars
            prev_history_chars_compresh = compresh_history_chars

            record = {
                "conversation_id": conversation.get("conversation_id"),
                "compresh_mode": compresh_mode,
                "block_label": block_label,
                "turn_idx": turn_idx,
                "user_msg_idx": user_turns_done,
                "user_message": content,
                "history_chars_raw": raw_history_chars,
                "history_chars_compresh": compresh_history_chars,
                "history_chars_raw_delta": history_chars_raw_delta,
                "history_chars_compresh_delta": history_chars_compresh_delta,
                "saving_pct": round(saving_pct, 2),
                "n_compressed_entries": sum(
                    len(b.compressed_refs) for b in prior_boxes
                ),
                "q_distribution_this_turn": q_distribution_this_turn,
                "q3_dedup_hits_this_turn": q3_dedup_hits_this_turn,
                "simulated_context_window": simulate_context_window,
                "raw_truncated_count": raw_truncated_count,
                "raw": {**raw_result, "error": raw_err},
                "compresh": {**compresh_result, "error": compresh_err},
                "provider": provider,
                "model": model,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            logger.info(
                "[T%d user#%d %s] saving=%.1f%% raw_in=%d compresh_in=%d "
                "Δraw=%d Δcompresh=%d retrieval=%d/%d",
                turn_idx, user_turns_done, compresh_mode, saving_pct,
                raw_result["tokens_in"], compresh_result["tokens_in"],
                history_chars_raw_delta, history_chars_compresh_delta,
                compresh_result["retrieval_ok"], compresh_result["retrieval_calls"],
            )
            time.sleep(throttle_s)

    logger.info("done — %d user turns processed", user_turns_done)
    logger.info("workdir (debug): %s", workdir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--conversation", required=True,
                   help="Path to lmsys_longest.json (output of select_longest.py)")
    p.add_argument("--provider", required=True,
                   choices=["openai", "anthropic", "openrouter"])
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-user-turns", type=int, default=None,
                   help="Cap number of user turns (default: all)")
    p.add_argument("--throttle-s", type=float, default=0.2)
    p.add_argument("--compresh-mode", choices=["tulbase", "tul1"],
                   default="tulbase",
                   help="Compression mode: 'tulbase' = format-cleaned "
                        "baseline (no Q matrix, no dedup); 'tul1' = full "
                        "TUL 1.0 (Q matrix + semantic store dedup).")
    p.add_argument("--simulate-context-window", type=int, default=None,
                   help="Fas 1c — simulate a smaller context window by "
                        "truncating raw mode's history to fit in N "
                        "tokens (approx, 1 token ≈ 4 chars). Compresh "
                        "history is left untouched. Use to compare same "
                        "model across budget levels (e.g. 8K, 16K, 32K).")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Quiet noisy dependencies unless --verbose. Bench progress lives
    # entirely in the runner's own logger; pipeline / HTTP client info
    # streams overwhelm the terminal for 288-turn runs.
    if not args.verbose:
        for noisy in (
            "httpx", "httpcore",
            "anthropic", "anthropic._base_client",
            "openai", "openai._base_client",
            "tulbase", "tulbase.pipeline",
            "tulbase.backfill", "tulbase.semantic_store",
            "tulbase.compression_log", "tulbase.modality",
            "tulbase.cold_storage", "tulbase.retrieval",
            "tulbase.summarizer",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set")
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")
    if args.provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("ERROR: OPENROUTER_API_KEY not set")

    convo = json.loads(Path(args.conversation).read_text(encoding="utf-8"))
    run(
        convo,
        provider=args.provider,
        model=args.model,
        output_path=Path(args.output),
        max_user_turns=args.max_user_turns,
        throttle_s=args.throttle_s,
        compresh_mode=args.compresh_mode,
        simulate_context_window=args.simulate_context_window,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
