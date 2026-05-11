# Changelog

All notable changes to tulbase are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-05-11

### Renamed
- Package renamed from **pith** to **tulbase**. The repo
  `compresh/compresh` (formerly `pith`) has been renamed to
  `compresh/tulbase`; the `pith` PyPI package is deprecated and
  receives no updates.

### Added — depth-aware context compression
- **Pipeline + TurnBox**: turn-by-turn marker format with epistemic
  transparency. Each conversation turn becomes a `[T<n> (Speaker)]`
  block with a short summary + `Compressed: ID=compr-...` references.
- **Cold storage + CompressionLog (DuckDB)**: SHA-256-deduped content
  store with on-demand `fetch_compressed(id)` retrieval.
- **Modality classifier**: deterministic regex segmentation (code,
  terminal output, JSON dumps, stack traces).
- **Tier-1 LLM-free summarizer**: LexRank-based, ~30-50 token budget.
- **Backfill mode**: mid-conversation activation with an accumulator
  for long-distance carry / resolves chains.
- **Provenance (source-aware speaker)**: 6-channel Channel enum +
  4-level TrustLevel for "who said it, when, from where" attribution.
- **Honesty system prompt** (mini & full variants): tells the model
  that compressed markers indicate honestly elided content; the
  model should call `fetch_compressed` for specifics and abstain
  cleanly when retrievable=false.
- **OpenAI + Anthropic tool definitions**: drop-in `fetch_compressed`
  and `list_compressed` tool surface.

### Added — proxy
- FastAPI router with OpenAI-compatible `/v1/chat/completions`.
- Auto-detect provider routing (`gpt-*` → OpenAI, `claude-*` →
  Anthropic, `org/model` → OpenRouter).
- Stateless per-request compression (temp DuckDB + cleanup).
- DeBERTa-v3 prompt injection detection (3-layer: regex + heuristic + ML).

### Added — bench infrastructure
- Multi-conv LMSYS-Chat-1M dataset builder (5-block concat).
- Turn-by-turn runner with raw vs compresh dual-context inference.
- Groq Llama 70B judge with MiniLM cosine similarity backup.
- Aggregate report generator (marginal cost curve + drift transition
  retention).
- Reproducible bench scripts (`bench/run_fas1a.sh`).

### Bench results
- **Claude Haiku 4.5**: 27.6 % cost saving, 67.1 % equivalence rate.
- **gpt-4o-mini**: 43.4 % cost saving, 65.7 % equivalence rate.
- Saving grows with conversation depth; equivalence stays
  approximately constant.

### What is NOT in 0.2.0
- TUL 1.0 upper layers (Q matrix episodic × affective classification,
  semantic_store Q3 dedup, directed forgetting) are part of the
  [Compresh](https://compre.sh) proprietary distribution. Patent
  pending (TR-TPMK 2026/007305) — those layers will ship under
  Apache 2.0 once the patent is granted (~November 2027).
- Aftermarket integrations (Cursor extension, VS Code, LangChain
  wrapper, MCP server, agent skills) moved to separate repos.

### Removed (from pith v0.1.0)
- `pith.optimizer` (936 lines) — Phase 2 distill / mesh / tag cloud
  logic archived. Replaced by turn-box + Tier-1 summarizer.
- `pith.distill`, `dual_encoder`, `mesh_store`, `belief_network`,
  `tag_extractor`, etc. — the entire Phase 2 paradigm is gone.
- LLMLingua-2 dependency — Tier-1 LLM-free pipeline is enough.
- Aftermarket templates (`extensions/`, `skills/`) — moved to
  separate repos.

### License
- Changed from Apache-2.0 to **MIT**. Patent-free core, more
  permissive license for community adoption. Upcoming TUL 1.0
  layers will be Apache 2.0 (patent grant version) when released.

## [0.1.0] — 2026-04-19 (pith)

Initial `pith` release: Phase 2 distill, tag cloud, mesh paradigm.
Archived in the `backup/pith-v0.1.0` branch of this repo. Superseded
by tulbase v0.2.0.
