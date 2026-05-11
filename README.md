# tulbase

> **Depth-aware context compression for LLM proxies.**
> Turn-box marker format with epistemic transparency, LLM-free Tier-1
> summarization, and on-demand `fetch_compressed` retrieval.

```bash
pip install tulbase
tulbase serve
```

Drop-in replacement for any OpenAI-compatible API. Point your LLM
client at `http://localhost:8000/v1` instead of `api.openai.com` —
tulbase compresses conversation history before forwarding, returns the
provider's response unchanged.

---

## What it does

### Turn-box context compression
Each conversation turn is wrapped in a structured marker block:
Code blocks, terminal output, JSON dumps, and stack traces are elided
into cold storage. The model sees a compact summary plus a marker; it
can call `fetch_compressed(id)` to pull the original content when it
needs the specifics.

### Honest forgetting
Markers tell the model **what was elided and why**. The model fetches
when it needs detail and abstains when retrievable=false. No
hallucinated content from compressed history.

### Structural injection protection
- DeBERTa-v3 ML classifier + 3-layer regex/heuristic detection
- TurnBox format isolates user messages and tool outputs in their own
  marker blocks — "I am the system" style injection vectors break
  structurally before reaching the model

---

## Benchmarks

Multi-conv LMSYS-Chat-1M bench, 288 messages across code-heavy,
persona, financial, opinion, and debugging content:

| Model | Mode | Cost saving | Equivalence rate |
|---|---|---|---|
| Claude Haiku 4.5 | tulbase | **27.6 %** | 67.1 % |
| gpt-4o-mini | tulbase | **43.4 %** | 65.7 % |

Saving grows with conversation depth; equivalence stays approximately
constant. See [`bench/`](bench/) for the full report and reproduction
recipe.

---

## Quickstart

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # tulbase proxy
    api_key="sk-...",                      # your real API key (forwarded)
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)
```

---

## Architecture
---

## What is NOT in tulbase

Tulbase is the **open-source core** of the Compresh stack. Higher-level
TUL 1.0 layers (episodic × affective classification, source-aware
provenance memory, semantic dedup, directed forgetting) are part of
the [Compresh](https://compre.sh) product but **not part of this
library**. Those will be Apache 2.0 licensed upon patent grant
([TR-TPMK 2026/007305](https://compre.sh/docs/patent), publication
Kasım 2027).

---

## Migration from `pith`

This package replaces the `pith` PyPI package (v0.1.0, deprecated).
The `pip install pith` distribution remains on PyPI as historical
artifact but receives no updates. Use `pip install tulbase` for new
work.

API rename:
```python
# Old
from tulbase import Pipeline
# New
from tulbase import Pipeline
```

See [CHANGELOG.md](CHANGELOG.md) for the full v0.1 → v0.2 migration.

---

## License

[MIT](LICENSE) — Copyright © 2026 [Compresh Ltd](https://compre.sh)

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
