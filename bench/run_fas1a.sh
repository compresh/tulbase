#!/usr/bin/env bash
#
# Fas 1a — multi-model fits-context bench
#
# Runs the same multi-conv dataset across 8 models (in addition to the
# already-completed Haiku + mini), each in both tulbase and tul1 modes.
# Models ordered by ascending cost so any early failure surfaces cheap.
#
# Required env vars (source proxy/.env first):
#   OPENROUTER_API_KEY
#
# Usage:
#   cd ~/projects/compresh-workspace
#   source compresh/migration-staging/proxy/.venv/bin/activate
#   set -a && source compresh/migration-staging/proxy/.env && set +a
#   bash compresh/benchmark/turn-by-turn/run_fas1a.sh
#
# Resumable: existing output files are skipped. Delete a file to re-run it.

set -u  # no -e: one model failure shouldn't abort the rest

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$ROOT/benchmark/turn-by-turn/data/lmsys_multi_conv.json"
RESULTS="$ROOT/benchmark/turn-by-turn/results"
TS=$(date +%Y%m%d)

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set. Source proxy/.env first." >&2
    exit 1
fi

mkdir -p "$RESULTS"

# (provider, model, label, throttle_s)
# Ordered by ascending cost — cheap models first so failures surface early.
# Models tagged as `openrouter` use OpenRouter's catalog. Model IDs are
# OpenRouter's canonical names (verify at https://openrouter.ai/models).
MODELS=(
    # IDs verified against OpenRouter catalog 2026-05-11.
    # Llama 3.1 8B intentionally kept at its 16K context window — bench
    # raw_in averages 17K, so raw mode will overflow ~T80 onward. That is
    # the "context overflow protection" demo we want: raw fails while
    # Compresh continues. Aggregate skips errored turns for that model
    # in its raw/equivalence stats.
    #
    # Qwen 2.5 7B excluded: OpenRouter's provider stack returns 400 on
    # tool-use calls (see compresh-side error log 2026-05-11 21:42).
    # A "tool-less Compresh" mode (Fas 1b candidate) would let it back in.
    "openrouter|mistralai/ministral-8b-2512|ministral-8b|0.3"
    "openrouter|meta-llama/llama-3.1-8b-instruct|llama-3.1-8b|0.3"
    "openrouter|google/gemini-2.5-flash|gemini-2.5-flash|0.3"
    "openrouter|deepseek/deepseek-chat|deepseek-v3|0.5"
    "openrouter|moonshotai/kimi-k2|kimi-k2|0.5"
    "openrouter|deepseek/deepseek-r1|deepseek-r1|0.5"
    "openrouter|meta-llama/llama-3.3-70b-instruct|llama-3.3-70b|0.5"
)

run_one() {
    local provider="$1"
    local model="$2"
    local label="$3"
    local mode="$4"
    local throttle="$5"

    local out="$RESULTS/multi-${label}-${mode}-${TS}.jsonl"
    if [[ -f "$out" ]]; then
        echo ">>> SKIP $label / $mode (already exists: $out)"
        return 0
    fi

    echo
    echo "============================================================"
    echo ">>> $label / $mode   (provider=$provider, throttle=${throttle}s)"
    echo "============================================================"
    local start=$(date +%s)

    python "$ROOT/benchmark/turn-by-turn/runner.py" \
        --conversation "$DATA" \
        --provider "$provider" \
        --model "$model" \
        --compresh-mode "$mode" \
        --throttle-s "$throttle" \
        --output "$out"
    local rc=$?

    local elapsed=$(( $(date +%s) - start ))
    if [[ $rc -ne 0 ]]; then
        echo ">>> FAILED $label / $mode (exit=$rc, ${elapsed}s)"
        # Keep partial output, may still be useful.
    else
        echo ">>> DONE $label / $mode (${elapsed}s)"
    fi
    return 0  # never abort the suite
}

for entry in "${MODELS[@]}"; do
    IFS='|' read -r provider model label throttle <<< "$entry"
    run_one "$provider" "$model" "$label" "tulbase" "$throttle"
    run_one "$provider" "$model" "$label" "tul1" "$throttle"
done

echo
echo "============================================================"
echo "Fas 1a complete. Results in: $RESULTS/"
ls -la "$RESULTS"/multi-*-${TS}.jsonl 2>/dev/null | grep -v judged
