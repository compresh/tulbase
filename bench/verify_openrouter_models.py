"""Verify OpenRouter model IDs before running the bench.

The per-model `/models/<id>` endpoint returns 404 in OpenRouter's
catalog, so we fetch the full `/api/v1/models` list and check each
target ID against the snapshot. Surfaces context_length to flag
overflow risk for the multi-conv dataset (max raw input ~26K chars
≈ 7K tokens).

Usage:

    set -a && source compresh/migration-staging/proxy/.env && set +a
    python compresh/benchmark/turn-by-turn/verify_openrouter_models.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

# Targets must match run_fas1a.sh.
TARGETS = [
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/ministral-8b-2512",
    "meta-llama/llama-3.1-8b-instruct",   # 16K — intentional overflow demo
    "google/gemini-2.5-flash",
    "deepseek/deepseek-chat",
    "moonshotai/kimi-k2",
    "deepseek/deepseek-r1",
    "meta-llama/llama-3.3-70b-instruct",
]

# Bench's heaviest raw turn is around 26K characters. Anything below
# ~25K context will overflow on the late turns — flag as overflow risk.
OVERFLOW_THRESHOLD_TOKENS = 25_000


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set. Source proxy/.env first.")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        sys.exit(f"ERROR: catalog fetch failed: {e}")

    all_models = {m["id"]: m for m in data.get("data", [])}
    print(f"Catalog has {len(all_models)} models")
    print()
    print(f"{'ID':<50s} {'ctx':>8s}  {'in$/M':>8s}  {'out$/M':>8s}  status")
    print("-" * 95)

    fail = 0
    overflow_risk = 0
    for tid in TARGETS:
        if tid not in all_models:
            print(f"{tid:<50s} {'—':>8s}  {'—':>8s}  {'—':>8s}  ✗ NOT FOUND")
            fail += 1
            continue
        m = all_models[tid]
        ctx = m.get("context_length", 0) or 0
        pricing = m.get("pricing", {}) or {}
        in_raw = pricing.get("prompt", "?")
        out_raw = pricing.get("completion", "?")

        def fmt_price(p):
            try:
                return f"${float(p)*1e6:.2f}"
            except (TypeError, ValueError):
                return str(p)

        in_str = fmt_price(in_raw)
        out_str = fmt_price(out_raw)

        if ctx < OVERFLOW_THRESHOLD_TOKENS:
            flag = f"⚠ overflow @ ~T{max(1, ctx//200)}"
            overflow_risk += 1
        else:
            flag = "✓ OK"

        print(f"{tid:<50s} {ctx:>8d}  {in_str:>8s}  {out_str:>8s}  {flag}")

    print()
    if fail > 0:
        print(f"FAILED: {fail} model(s) not found in catalog.")
        print("Browse https://openrouter.ai/models to find the correct ID,")
        print("then update run_fas1a.sh.")
        return 1

    if overflow_risk > 0:
        print(f"NOTE: {overflow_risk} model(s) below {OVERFLOW_THRESHOLD_TOKENS} "
              "ctx — will overflow on the late turns of the multi-conv "
              "dataset.")
        print("This is INTENTIONAL for the 'context overflow protection' "
              "demo case (Compresh continues while raw fails).")
        print("Aggregate skips raw-errored turns for those models.")

    print()
    print("All models found — safe to run run_fas1a.sh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
