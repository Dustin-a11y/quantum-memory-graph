# Quantum Memory Graph ⚛️🧠

[![PyPI version](https://img.shields.io/pypi/v/quantum-memory-graph)](https://pypi.org/project/quantum-memory-graph/)
[![PyPI downloads](https://img.shields.io/pypi/dm/quantum-memory-graph)](https://pypi.org/project/quantum-memory-graph/)
[![LongMemEval](https://img.shields.io/badge/LongMemEval-%231-94.26%25_NDCG-brightgreen)](https://github.com/xiaowu0162/LongMemEval/issues/46)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-ghcr.io-blue)](https://github.com/users/Dustin-a11y/packages/container/package/quantum-memory-graph)

**Relationship-aware memory for AI agents. Knowledge graphs + quantum-optimized subgraph selection.**

Every memory system treats memories as independent documents — search, rank, stuff into context. But memories aren't independent. They have *relationships*. "The team chose React" becomes 10x more useful paired with "because of ecosystem maturity" and "FastAPI handles the backend."

## 🏆 #1 on LongMemEval (ICLR 2025 Benchmark)

Tested on the official [LongMemEval benchmark](https://arxiv.org/abs/2410.10813) — [verified submission](https://github.com/xiaowu0162/LongMemEval/issues/46).

| Method | R@1 | R@5 | R@10 | NDCG@10 |
|--------|:---:|:---:|:----:|:-------:|
| OMEGA (prev SOTA) | — | 89.2% | 94.1% | 87.5% |
| Mastra OM | — | 91.0% | 95.2% | 89.1% |
| **QMG v1.1 (published #1)** | — | **95.8%** | **98.85%** | **93.2%** |
| **QMG v1.2 — chunked retrieval pipeline** 🏆 | **90.6%** | **98.6%** | **99.4%** | **94.26%** |
| **QMG v1.3 — +BM25 hybrid retrieval** 🥇 | 90.0% | **99.0%** | **99.8%** | 93.32% |

**Competitor comparison (same benchmark):**

| System | R@5 | Source |
|--------|:---:|--------|
| **QMG v1.3 (BM25 hybrid)** | **99.0%** | This repo — `benchmarks/longmemeval_bm25_hybrid_results.json` |
| QMG v1.2 (chunked gte-large) | 98.6% | This repo |
| Mem0 (Apr 2026) | 94.8% | [mem0.ai/research](https://mem0.ai/research) |
| Mastra OM | 91.0% | [LongMemEval #46](https://github.com/xiaowu0162/LongMemEval/issues/46) |
| OMEGA (prev SOTA) | 89.2% | LongMemEval paper |

**Benchmark run:** 500 questions, chunked gte-large embeddings (500-char blocks, 100-char overlap, mean-of-top-3 session scoring). Verified on DGX Spark GB10 (CUDA, ~53 min).

**Chunking technique:** Each session split into overlapping 500-char chunks → gte-large embedding → per-session score = mean of top-3 chunk scores → rank by score. This recovers the v7 methodology that achieved our original #1, now verified with a clean reproducible pipeline.

**BM25 hybrid (v1.3):** Keyword matching (BM25) fused with embedding scores at 70/30 ratio using stopword-filtered tokenization. Provides +0.4% R@5 lift at the ceiling — significant when every miss counts. The `rank_bm25` package is optional — falls back to embedding-only if not installed.

**See:** `benchmarks/run_longmemeval_chunked_staged.py` and `benchmarks/run_longmemeval_hybrid.py` for exact benchmark code. `benchmarks/longmemeval_bm25_hybrid_results.json` for full per-question results.

**⚠️ Metric note:** The retrieval benchmarks above use `recall_any` (whether at least one answer session appears in the top-K). The official `evaluate_qa.py` strict parser uses `recall_all` which is a stricter metric — see the end-to-end QA section below.

## 🤖 Native End-to-End QA — 85.6% (Official Pipeline)

In addition to retrieval-only benchmarks, QMG has been evaluated end-to-end on the official LongMemEval pipeline: retrieval → hypothesis generation → GPT-4o judge. **This is a self-run evaluation on the official dataset and codebase — not authored or verified by the LongMemEval maintainers.**

| Metric | Value |
|--------|:-----:|
| **Overall QA Accuracy** | **85.6%** (428/500) |
| Generation model | DeepSeek R1 (OpenRouter) |
| Judge | GPT-4o (OpenRouter) |
| Judge cost | $0.55 |
| Errors | 0 |

**By question type:**

| Type | Accuracy | Items |
|------|:--------:|:-----:|
| single-session-user | 98.57% | 70 |
| single-session-assistant | 96.43% | 56 |
| knowledge-update | 87.18% | 78 |
| temporal-reasoning | 81.95% | 133 |
| single-session-preference | 80.00% | 30 |
| multi-session | 78.20% | 133 |

**Retrieval metrics (official strict parser, recall_all):** recall_all@5 0.8723, ndcg_any@5 0.8993, recall_all@10 0.9511, ndcg_any@10 0.9176.

**Full artifacts** including hypotheses, per-item eval results, scripts, tests, and provenance: [`benchmarks/longmemeval/native-official-20260716/`](./benchmarks/longmemeval/native-official-20260716/PROVENANCE.md)

**Separate from Bench'd.ai:** The Bench'd.ai score (86.2) uses a different harness. This native run uses the exact official LongMemEval repository, `prepare_prompt`, and `get_anscheck_prompt` with zero modifications. See the [official issue #46](https://github.com/xiaowu0162/LongMemEval/issues/46) for the submission record.

## Install

```bash
# Python (all platforms)
pip install quantum-memory-graph

# macOS
brew tap Dustin-a11y/qmg && brew install quantum-memory-graph

# Docker
docker pull ghcr.io/dustin-a11y/quantum-memory-graph:latest

# Node.js (thin wrapper)
npm install -g qmg

# Conda (pending — PR #33723)
conda install -c conda-forge quantum-memory-graph
```

## Quick Start

```python
from quantum_memory_graph import store, recall

# Store memories — automatically builds knowledge graph
store("Project Alpha uses React frontend with TypeScript.")
store("Project Alpha backend is FastAPI with PostgreSQL.")
store("FastAPI connects to PostgreSQL via SQLAlchemy ORM.")
store("React components use Material UI for styling.")
store("Team had pizza for lunch. Pepperoni was great.")

# Recall — graph traversal + QAOA finds the optimal combination
result = recall("What is Project Alpha's full tech stack?", K=4)

for memory in result["memories"]:
    print(f"  {memory['text']}")
    print(f"    Connected to {len(memory['connections'])} other selected memories")
```

Output: Returns React, FastAPI, PostgreSQL, and SQLAlchemy memories — connected, complete, no noise. The pizza memory is excluded because it has no graph connections to the tech stack cluster.

## How It Works

```
Query: "What's the tech stack?"
        │
        ▼
┌─────────────────────┐
│  1. Hybrid Search     │  BM25 keyword + embedding cosine (70/30 fusion)
│     Find neighbors   │  Discovers memories connected to relevant ones
└────────┬────────────┘
         │ 14 candidates
         ▼
┌─────────────────────┐
│  2. Subgraph Data    │  Extract adjacency matrix + relevance scores
│     Build problem    │  Encode relationships as optimization weights
└────────┬────────────┘
         │ NP-hard selection
         ▼
┌─────────────────────┐
│  3. QAOA Optimize    │  Quantum approximate optimization
│     Find best K      │  Maximizes: relevance + connectivity + coverage
└────────┬────────────┘
         │ K memories
         ▼
┌─────────────────────┐
│  4. Return with      │  Each memory includes its connections
│     relationships    │  to other selected memories
└─────────────────────┘
```

### Why Quantum?

Optimal subgraph selection is NP-hard. Given N candidate memories, finding the best K that maximize relevance, connectivity, AND coverage has exponential classical complexity. QAOA provides polynomial-time approximate solutions that beat greedy heuristics — this is the one problem where quantum computing has a genuine algorithmic advantage over classical approaches.

## Architecture

### Three Layers

1. **Knowledge Graph** (`graph.py`) — Memories are nodes. Relationships are weighted edges based on:
   - Semantic similarity (embedding cosine distance)
   - BM25 keyword matching (70/30 hybrid fusion)
   - Entity co-occurrence (shared people, projects, concepts)
   - Temporal proximity (memories close in time)
   - Source proximity (same conversation/document)

2. **Subgraph Optimizer** (`subgraph_optimizer.py`) — QAOA circuit that maximizes:
   - α × relevance (individual memory scores from hybrid BM25+embedding)
   - β × connectivity (edge weights within selected subgraph)
   - γ × coverage (topic diversity across selection)

3. **Pipeline** (`pipeline.py`) — Unified `store()` and `recall()` interface.

```

## API Server

```bash
pip install quantum-memory-graph[api]
python -m quantum_memory_graph.api
```

Endpoints:
- `POST /store` — Store a memory
- `POST /recall` — Graph + QAOA recall
- `POST /store-batch` — Batch store
- `GET /stats` — Graph statistics
- `GET /` — Health check

## Advanced Usage

### Custom Graph

```python
from quantum_memory_graph import MemoryGraph, recall
from quantum_memory_graph.pipeline import set_graph

# Tune similarity threshold for edge creation
graph = MemoryGraph(similarity_threshold=0.25)
set_graph(graph)

# Store and recall as normal
```

### Tune QAOA Parameters

```python
result = recall(
    "query",
    K=5,
    alpha=0.4,       # Relevance weight
    beta_conn=0.35,   # Connectivity weight  
    gamma_cov=0.25,   # Coverage/diversity weight
    hops=3,           # Graph traversal depth
    top_seeds=7,      # Initial seed nodes
    max_candidates=14, # Max qubits for QAOA
)
```

```python

def my_recall(memories, query, K):
    # Your recall implementation
    return selected_indices  # List[int]

results = run_benchmark(my_recall, K=5)
print(f"Coverage: {results['avg_coverage']*100:.1f}%")
```

## IBM Quantum Hardware

For production workloads, run QAOA on real quantum hardware:

```bash
pip install quantum-memory-graph[ibm]
export IBM_QUANTUM_TOKEN=your_token
```

Validated on `ibm_fez` and `ibm_kingston` backends.

## Requirements

- Python ≥ 3.9
- sentence-transformers
- networkx
- qiskit + qiskit-aer
- numpy

## License

MIT License — Copyright 2026 Coinkong (Chef's Attraction)

## Community Integrations

### Hermes Agent — QMG Memory Provider

[`smoke-ui/hermes-qmg`](https://github.com/smoke-ui/hermes-qmg) (v2.0.0) — A community-maintained [Hermes Agent](https://github.com/nousresearch/hermes-agent) memory provider that integrates QMG as a backend for the Hermes MemoryProvider API, using graph recall + QAOA subgraph selection for relationship-aware agent memory.

- **Install:** `hermes plugin install smoke-ui/hermes-qmg`
- **Status:** Independently audited — 35/35 tests passing, ruff-clean, PyPI dependency hash verified
- **Maintainer:** Community-maintained, independently developed — not affiliated with the QMG project or Nous Research
- **⚠️ Inspect before running** — This is a community plugin; review the code before installing. No official endorsement by the QMG project or Nous Research.

> A [collaboration issue](https://github.com/smoke-ui/hermes-qmg/issues/1) is open to coordinate future development between the plugin and QMG.

## Links

- [GitHub](https://github.com/Dustin-a11y/quantum-memory-graph) — Source code and benchmarks
- [mem0 vs QMG Comparison](./COMPARISON.md) — How QMG differs from the incumbent
- [Hermes Agent Case Study](./CASE_STUDY_HERMES.md) — QMG in production with 12+ agents
- [LongMemEval Submission](https://github.com/xiaowu0162/LongMemEval/issues/46) — Verified #1 ranking
- [Conda-forge PR](https://github.com/conda-forge/staged-recipes/pull/33723) — Track conda availability
