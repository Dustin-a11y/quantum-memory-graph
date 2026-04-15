# Quantum Memory Graph ⚛️🧠

**Full memory system for AI agents. Knowledge graphs + QAOA optimization + semantic tiers + deduplication + cross-agent sharing.**

v1.0.0 — the complete memory architecture for multi-agent systems.

## What It Does

Most memory systems treat memories as independent documents — search, rank, stuff into context. QMG maps **relationships** between memories, organizes them into **tiers** by recency, **deduplicates** similar memories, and enables **cross-agent knowledge sharing**. The QAOA optimizer selects the best *combination* of memories — not just the most relevant individuals, but the best connected subgraph.

## Architecture

```
                    ┌─────────────────────────────┐
                    │     Hot Tier (< 1 hour)      │  In-memory deque
                    │  Zero latency, auto-demote   │  Survives compaction
                    └──────────┬──────────────────┘
                               │ demote
                    ┌──────────▼──────────────────┐
                    │    Warm Tier (1–24 hours)     │  SQLite-backed
                    │  Keyword searchable, fast     │  Per-agent isolation
                    └──────────┬──────────────────┘
                               │ demote
                    ┌──────────▼──────────────────┐
                    │   Cold Tier (> 24 hours)      │  Full knowledge graph
                    │  QAOA optimization, semantic   │  Embedding search
                    └──────────┬──────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
  ┌───────▼───────┐  ┌────────▼────────┐  ┌───────▼───────┐
  │ Dedup Engine   │  │ Shared Memory   │  │ Obsidian      │
  │ Cosine ≥ 0.95  │  │ Cross-agent     │  │ Vault Export  │
  │ Smart merge    │  │ Access control  │  │ Wikilinks     │
  └───────────────┘  └─────────────────┘  └───────────────┘
```

## Benchmarks

### LongMemEval (ICLR 2025) — Industry Standard

| System | R@5 | R@10 | NDCG@10 |
|--------|-----|------|---------|
| **Quantum Memory Graph (gte-large)** | **96.6%** | **98.7%** | **94.3%** |
| OMEGA | 95.4% | — | — |
| Mastra OM | 94.9% | — | — |
| Quantum Memory Graph (MiniLM, default) | 93.4% | 97.4% | 90.8% |

**#1 to our knowledge.** Tied on R@5, best R@10 and NDCG@10 among published results.

### MemCombine — Combination Recall (250 Scenarios)

| Method | Coverage | Evidence Recall | F1 | Perfect |
|--------|----------|----------------|----|---------|
| Embedding Top-K | 92.3% | 93.9% | 91.3% | 181/250 |
| Graph + QAOA (default) | 94.6% | 95.9% | 93.5% | 210/250 |
| **Graph + QAOA (tuned)** | **97.0%** | **98.6%** | **96.0%** | **215/250** |

Optimal weights (grid search, 320 combos): `α=0.3, β=0.15, γ=0.35, threshold=0.25`

**+4.7% over Top-K baseline. +2.5% over default weights.**

## Install

```bash
# Core (no quantum deps, uses greedy fallback)
pip install quantum-memory-graph

# With QAOA quantum optimization
pip install quantum-memory-graph[quantum]

# Full install (API server + quantum + NLP)
pip install quantum-memory-graph[full]

# Development
pip install quantum-memory-graph[dev]
```

## Quick Start

```python
from quantum_memory_graph import store, recall

# Store memories — builds knowledge graph automatically
store("Project Alpha uses React with TypeScript.")
store("Project Alpha backend is FastAPI with PostgreSQL.")
store("FastAPI connects to PostgreSQL via SQLAlchemy ORM.")

# Recall — graph traversal + QAOA finds the optimal combination
result = recall("What is Project Alpha's tech stack?", K=4)
for memory in result["memories"]:
    print(f"  {memory['text']}")
```

## Features

### 1. Semantic Memory Tiers

Three-tier system for optimal latency at every time scale:

```python
from quantum_memory_graph.tiers import MemoryTierManager

tiers = MemoryTierManager(agent_id="daisy", warm_db_path="~/.qmg/warm.db")

# Store — goes to hot tier automatically
tiers.store("User wants the dashboard updated", agent_id="daisy")

# Recall — searches all tiers, hot first
results = tiers.recall("dashboard", agent_id="daisy", limit=5)

# Maintenance — demotes hot → warm → cold
stats = tiers.tick()
print(stats)  # {"hot": 5, "warm": 23, "cold": 412, "demoted": 3}
```

- **Hot** (< 1 hour): In-memory deque. Zero latency. Survives context compaction.
- **Warm** (1–24 hours): SQLite-backed. Keyword searchable. Per-agent isolation.
- **Cold** (> 24 hours): Full knowledge graph + QAOA optimization.

### 2. Memory Deduplication

Keeps your memory clean — merges near-duplicates automatically:

```python
from quantum_memory_graph.dedup import MemoryDeduplicator

dedup = MemoryDeduplicator(threshold=0.95)

# Dry run first
stats = dedup.merge_duplicates(graph, dry_run=True)
print(f"Would remove {stats['duplicates_removed']} duplicates")

# Real merge — keeps the richest version of each memory
stats = dedup.merge_duplicates(graph)
print(f"Removed {stats['duplicates_removed']}, merged {stats['entities_merged']} entities")
```

Smart canonical selection: keeps the memory with the most entities, most recent timestamp, and longest text.

### 3. Cross-Agent Memory Sharing

Shared knowledge pool with access control:

```python
from quantum_memory_graph.sharing import SharedMemoryPool

pool = SharedMemoryPool(db_path="~/.qmg/shared_pool.db")

# Store shared knowledge — visible to specific agents or all
pool.store(
    text="Chef's Attraction runs on CookUnity",
    author_agent="daisy",
    category="business",
    access="public"  # or ["daisy", "luigi", "bowser"]
)

# Any authorized agent can recall
results = pool.recall("CookUnity", requesting_agent="mario", limit=5)
```

Categories: `business`, `technical`, `rules`, `people`, `general`

### 4. Obsidian Vault Export

Visualize your agent's knowledge graph in Obsidian:

```python
from quantum_memory_graph.obsidian import export_vault, export_from_mem0

# Export from a running QMG graph
export_vault(graph, "/path/to/vault", agent_memories={"daisy": [...], "dk": [...]})

# Or pull directly from a Mem0 API
export_from_mem0(
    mem0_url="http://localhost:8500",
    vault_path="/path/to/vault",
    agents=["daisy", "luigi", "bowser"]
)
```

Each memory becomes a markdown note with YAML frontmatter, `[[wikilinks]]` for connections, and `#tags` for entities. Open in Obsidian → Graph View to see your agent's knowledge mapped visually.

### 5. QAOA Subgraph Optimization

The core quantum advantage — optimal memory combination selection:

```python
result = recall(
    "query",
    K=5,
    alpha=0.4,        # Relevance weight
    beta_conn=0.35,    # Connectivity weight
    gamma_cov=0.25,    # Coverage/diversity weight
    hops=3,            # Graph traversal depth
    top_seeds=7,       # Initial seed nodes
    max_candidates=14, # Max candidates for QAOA
)
```

Falls back to greedy selection when Qiskit is not installed — still beats pure similarity search.

### 6. Short-Term Memory

Session-aware memory with recency boosting:

```python
from quantum_memory_graph.recency import ShortTermMemory

stm = ShortTermMemory()
stm.start_session("conv_123")
stm.add_turn("conv_123", "What's our deadline?", "March 15th")

# Recent conversations get boosted in recall
boosted = stm.boost_results(results, session_id="conv_123")
```

## API Server

```bash
pip install quantum-memory-graph[api]
python -m quantum_memory_graph.api --port 8502
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/store` | Store a memory |
| `POST` | `/store-batch` | Batch store (max 500) |
| `POST` | `/recall` | Graph + QAOA recall |
| `POST` | `/dedup` | Run deduplication |
| `POST` | `/tiers/store` | Store to tier system |
| `POST` | `/tiers/recall` | Recall from all tiers |
| `POST` | `/tiers/tick` | Run tier maintenance |
| `POST` | `/shared/store` | Store shared memory |
| `POST` | `/shared/recall` | Recall shared memories |
| `GET` | `/shared/stats` | Shared pool statistics |
| `POST` | `/obsidian/export` | Export to Obsidian vault |
| `GET` | `/stats` | Graph statistics |
| `GET` | `/` | Health check |

All endpoints require `Authorization: Bearer <token>` when `QMG_API_TOKEN` is set.

## Choosing a Model

| Model | Size | GPU? | R@5 | Best For |
|-------|------|------|-----|----------|
| `all-MiniLM-L6-v2` (default) | 90MB | No | 93.4% | Laptops, CI/CD |
| `BAAI/bge-large-en-v1.5` | 1.3GB | Recommended | 95.9% | Production with GPU |
| `intfloat/e5-large-v2` | 1.3GB | Recommended | 96.0% | Best ranking (NDCG) |
| `thenlper/gte-large` | 1.3GB | Recommended | 96.6% | Maximum accuracy |

```python
from quantum_memory_graph import MemoryGraph

mg = MemoryGraph()                              # Default — works everywhere
mg = MemoryGraph(model="thenlper/gte-large")    # Best accuracy
```

## IBM Quantum Hardware

For production QAOA on real quantum hardware:

```bash
pip install quantum-memory-graph[quantum]
export IBM_QUANTUM_TOKEN=your_token
```

Validated on `ibm_fez` and `ibm_kingston` backends.

## Requirements

- Python ≥ 3.9
- sentence-transformers, networkx, numpy
- Optional: qiskit + qiskit-aer (quantum), fastapi + uvicorn (API), spacy (NLP)

## License

MIT License — Copyright 2026 Coinkong (Chef's Attraction)

## Links

- [GitHub](https://github.com/Dustin-a11y/quantum-memory-graph)
- [PyPI](https://pypi.org/project/quantum-memory-graph/)
- [Changelog](CHANGELOG.md)

## Author

Built by **Dustin Taylor** / Coinkong — Chef's Attraction AI Lab

- 𝕏 [@Coinkong](https://x.com/Coinkong)
- 📸 [@Dustin.c.Taylor](https://instagram.com/Dustin.c.Taylor)
