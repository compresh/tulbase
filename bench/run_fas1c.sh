#!/usr/bin/env bash
#
# Fas 1c — controlled context-budget experiment
#
# Runs ONE model (Llama 3.3 70B by default — 131K real context, plenty
# of room for the simulation) across four simulated context windows.
# Truncation only applies to raw mode; compresh mode is left intact.
#
# The story this bench tells:
#   "As the raw context budget shrinks, raw answers degrade. Compresh
#    answers stay stable because the compressed history fits in any
#    budget."
#
# Required env vars (source proxy/.env first):
#   OPENROUTER_API_KEY
#
# Usage:
#   cd ~/projects/compresh-workspace
#   source compresh/migration-staging/proxy/.venv/bin/activate
#   set -a && source compresh/migration-staging/proxy/.env && set +a
#   bash compresh/benchmark/turn-by-turn/run_fas1c.sh

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$ROOT/benchmark/turn-by-turn/data/lmsys_multi_conv.json"
RESULTS="$ROOT/benchmark/turn-by-turn/results"
TS=$(date +%Y%m%d)

MODEL_ID="meta-llama/llama-3.3-70b-instruct"
MODEL_LABEL="llama-3.3-70b"
PROVIDER="openrouter"
THROTTLE=0.5

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set. Source proxy/.env first." >&2
    exit 1
fi

mkdir -p "$RESULTS"

# Simulated context budgets, in tokens. The multi-conv bench's late
# turns push raw input to ~7K tokens, so:
#   4K   → severe truncation, most turns will fail/degrade
#   8K   → moderate, late turns clipped
#   16K  → mild, only the heaviest turns clipped
#   full → no truncation, baseline
LEVELS=(4096 8192 16384 0)  # 0 = no truncation

run_one() {
    local budget="$1"
    local mode="$2"

    local budget_tag
    if [[ "$budget" == "0" ]]; then
        budget_tag="full"
    else
        budget_tag="${budget}"
    fi

    local out="$RESULTS/fas1c-${MODEL_LABEL}-ctx${budget_tag}-${mode}-${TS}.jsonl"
    if [[ -f "$out" ]]; then
        echo ">>> SKIP $budget_tag / $mode (already exists: $out)"
        return 0
    fi

    echo
    echo "============================================================"
    echo ">>> Fas 1c: ${MODEL_LABEL} ctx=${budget_tag} mode=${mode}"
    echo "============================================================"
    local start=$(date +%s)

    local sim_flag=""
    if [[ "$budget" != "0" ]]; then
        sim_flag="--simulate-context-window $budget"
    fi

    # shellcheck disable=SC2086
    python "$ROOT/benchmark/turn-by-turn/runner.py" \
        --conversation "$DATA" \
        --provider "$PROVIDER" \
        --model "$MODEL_ID" \
        --compresh-mode "$mode" \
        --throttle-s "$THROTTLE" \
        $sim_flag \
        --output "$out"
    local rc=$?

    local elapsed=$(( $(date +%s) - start ))
    if [[ $rc -ne 0 ]]; then
        echo ">>> FAILED ctx=$budget_tag / $mode (exit=$rc, ${elapsed}s)"
    else
        echo ">>> DONE ctx=$budget_tag / $mode (${elapsed}s)"
    fi
    return 0
}

for budget in "${LEVELS[@]}"; do
    run_one "$budget" "tulbase"
    run_one "$budget" "tul1"
done

echo
echo "============================================================"
echo "Fas 1c complete. Results in: $RESULTS/"
ls -la "$RESULTS"/fas1c-${MODEL_LABEL}-*-${TS}.jsonl 2>/dev/null
