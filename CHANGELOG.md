# Changelog

All notable changes to Quantum Memory Graph.

## [1.0.0] — 2026-04-13

### 🚀 Full Memory System — 5 Major Features

**1. Memory Deduplication Engine**
- Cosine similarity-based duplicate detection (configurable threshold, default 0.95)
- Smart canonical selection: keeps richest memory (most entities, most recent, longest text)
- Entity merging: duplicate entities merged into canonical before removal
- Dry-run mode for preview before merge
- `POST /dedup` API endpoint

**2. Semantic Memory Tiers (Hot/Warm/Cold)**
- **Hot tier**: In-memory deque, <1 hour, injected every message. Zero latency.
- **Warm tier**: SQLite-backed, 1-24 hours, keyword-searchable, survives restarts
- **Cold tier**: Full mem0/QMG store, >24 hours, recalled on demand
- Automatic tier demotion: hot→warm→cold via `tick()` maintenance
- Per-agent isolation in warm tier
- `POST /tiers/store`, `POST /tiers/recall`, `POST /tiers/tick` API endpoints

**3. Cross-Agent Memory Sharing**
- Shared knowledge pool with SQLite persistence
- Access control: public (all agents) or restricted (named agents only)
- Category tagging: business, technical, rules, people, general
- Author tracking and read counts
- `POST /shared/store`, `POST /shared/recall`, `GET /shared/stats` API endpoints

**4. Obsidian Vault Export**
- Export QMG graph as Obsidian-compatible markdown vault
- Each memory = one note with YAML frontmatter
- Graph edges = `[[wikilinks]]` between notes
- Entities = `#tags`
- Multi-agent support: one folder per agent
- Auto-generated `.obsidian` config with graph view settings
- `export_from_mem0()` — one-command pull from mem0 API + export
- `POST /obsidian/export` API endpoint

**5. Qiskit Graceful Fallback**
- QAOA optimizer now falls back to greedy selection when qiskit is not installed
- Enables lightweight deployments without quantum dependencies
- Full quantum mode still available when qiskit + qiskit-aer are present

### Infrastructure
- 44 tests, all passing (dedup: 8, tiers: 11, sharing: 8, recency: 17)
- Lazy imports for heavy dependencies (qiskit, sentence-transformers)
- SQLite for tier + sharing persistence (survives restarts)

## [0.4.0] — 2026-04-09

### Short-Term Memory Layer

- RecencyBooster: time-based decay boost for recent memories
- WorkingMemory: hot cache of last N memories (O(1) access)
- ConversationContext: current conversation tracking with memory boost
- ShortTermMemory: unified wrapper that plugs into recall pipeline
- 17 tests for all STM components

## [0.3.0] — 2026-04-08

### 🏆 #1 on LongMemEval

- **gte-large model: 96.6% R@5** — tied #1 worldwide on LongMemEval
- **98.7% R@10** — best in the world (MemPalace: 98.2%)
- **94.3% NDCG@10** — best ranking quality by 5.4 points
- Tested 5 embedding models in one day: MiniLM, BGE-large, e5-large, gte-large, nomic-embed
- Added `model=` parameter to `MemoryGraph()` for swappable embeddings
- Default stays MiniLM (90MB, no GPU, 93.4% R@5)

### Leaderboard

| System | R@5 | R@10 | NDCG@10 |
|--------|-----|------|---------|
| **Quantum Memory Graph (gte-large)** | **96.6%** | **98.7%** | **94.3%** |
| MemPalace | 96.6% | 98.2% | 88.9% |
| OMEGA | 95.4% | — | — |
| Mastra OM | 94.9% | — | — |

## [0.2.0] — 2026-04-08

### BGE-large support

- Added BGE-large-en-v1.5 as optional high-accuracy model (95.9% R@5)
- Published all benchmark scripts (v3–v6) and results
- LongMemEval v6 model comparison framework

## [0.1.0] — 2026-04-07

### Initial release

- Knowledge graph memory with QAOA-optimized subgraph selection
- MemCombine benchmark: 96.2% coverage (+3.8% over Top-K)
- LongMemEval baseline: 93.4% R@5 with chunked retrieval
- FastAPI server for production deployment
- IBM Quantum hardware support via qiskit-ibm-runtime
- Built on MemPalace by @bensig (MIT)

---

**Copyright 2026 Coinkong (Chef's Attraction AI Lab). MIT License.**
