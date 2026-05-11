# tulbase bench

Reproducible benchmark suite for depth-aware context compression.

## What's here

| File | Purpose |
|---|---|
| `runner.py` | Turn-by-turn raw vs compresh dual-context inference |
| `judge_groq.py` | Groq Llama judge with MiniLM cosine similarity |
| `aggregate.py` | Build comparison report from judged JSONL |
| `build_multi_conv.py` | Assemble multi-conv LMSYS-Chat-1M dataset |
| `select_longest.py` | Pick the longest English conversation |
| `verify_openrouter_models.py` | Sanity-check OpenRouter model IDs |
| `run_fas1a.sh` | One-command bench across 7 OpenRouter models |

## Quick start

```bash
# Build the 5-block multi-conv dataset
python bench/build_multi_conv.py --output bench/data/lmsys_multi_conv.json

# Verify your OpenRouter model IDs
python bench/verify_openrouter_models.py

# Run the multi-model bench (3-5 hours, ~$10 OpenRouter cost)
bash bench/run_fas1a.sh

# Judge + aggregate
for f in bench/results/multi-*-tulbase-*.jsonl; do
  python bench/judge_groq.py --pairs "$f" \
    --output "${f%.jsonl}-judged.jsonl"
done
python bench/aggregate.py \
  --judged-files bench/results/multi-*-judged.jsonl \
  --output bench/results/comparison.md
```

## Required env vars

- `OPENROUTER_API_KEY` — for the multi-model bench
- `GROQ_API_KEY` — for the judge (Llama 3.3 70B)
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — if running Anthropic /
  OpenAI directly

## Method

The bench uses sert (no drift bridge) concat of 5 LMSYS conversations
spanning code-heavy debugging, persona, financial analysis, opinion
rating, and JS debugging — 288 messages total. Every user turn is asked
twice: once with raw history, once with tulbase-compressed history. The
judge then rates whether the two answers are functionally equivalent.

Result: **~25-43 % cost saving across providers, equivalence rate stays
flat with depth.** Full numbers in [`results/`](results/).
