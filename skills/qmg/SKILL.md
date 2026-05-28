---
name: qmg
title: Quantum Memory Graph (QMG)
description: "Fleet-wide memory system with quantum-enhanced QAOA subgraph optimization — per-agent isolated memory graphs, cascade recall, and Pauli Correlation Encoding (PCE) for large-scale selection."
tags: [qmg, memory-graph, retrieval, qaoa, quantum, ibm, benchmarking]
---

# Quantum Memory Graph ⚛️🧠

**Relationship-aware memory for AI agents. Knowledge graphs + quantum-optimized subgraph selection.**

Every memory system treats memories as independent documents — search, rank, stuff into context. But memories aren't independent. They have *relationships*. QMG maps these relationships, then uses QAOA (Quantum Approximate Optimization Algorithm) to find the optimal *combination* of memories — not just the most relevant individuals, but the best connected subgraph.

## 🏆 Leaderboard Results

### LongMemEval (ICLR 2025) — #1

| Method | R@1 | R@5 | R@10 | NDCG@10 |
|--------|:---:|:---:|:----:|:-------:|
| OMEGA (prev SOTA) | — | 89.2% | 94.1% | 87.5% |
| Mastra OM | — | 91.0% | 95.2% | 89.1% |
| **QMG v1.2 (this repo)** 🏆 | **90.6%** | **98.6%** | **99.4%** | **0.9426** |

**Technique:** Chunked gte-large embeddings (500-char blocks, 100-char overlap, mean-of-top-3 session scoring). Verified on 500 questions.

### MemCombine — Memory Combination Quality

| Method | Coverage | Evidence Recall | F1 | Perfect |
|--------|----------|----------------|----|---------|
| Embedding Top-K | 69.9% | 65.6% | 68.1% | 1/5 |
| **Graph + QAOA** | **96.7%** | **91.0%** | **92.6%** | **4/5** |

## Architecture

### Per-Agent Isolation

Each agent gets their own QMG instance on a dedicated port with an isolated data directory. Zero cross-agent memory bleed.

```
Agent A ──→ QMG Instance A (port X, isolated data)
Agent B ──→ QMG Instance B (port Y, isolated data)
```

### Cascade Recall Flow

```
qmg_recall(query)
  Step 1: PERSONAL graph → results with relevance > 0.4?
    ├─ YES → return personal results
    └─ NO  → Step 2: HISTORICAL archive (shared, read-only fallback)
              └─ CACHE historical results into personal graph
```

### Three-Layer Architecture

1. **Knowledge Graph** — Memories as nodes, weighted edges (semantic, entity, temporal, source)
2. **Subgraph Optimizer** — QAOA circuit maximizing relevance + connectivity + coverage
3. **Pipeline** — Unified store/recall with auto-fallback

## Quantum Optimization

### Standard QAOA (≤14 candidates)
1-qubit-per-candidate encoding, up to 14 candidates on simulator.

### Pauli Correlation Encoding (PCE) (>14 candidates)
Compresses `m` candidates into O(√m) qubits:
- 20 candidates → 5 qubits (71% reduction)
- 100 candidates → 9 qubits (91% reduction)

Uses multi-basis measurement (X, Y, Z) + CVaR decode. Proven to beat greedy by 12.8% on adversarial landscapes.

### CVaR Subgraph Selection
Decodes QAOA output by taking the lowest-energy 10% of shots, computes per-candidate expectations. Finds globally optimal solutions on adversarial landscapes where greedy fails.

## Setup

```bash
pip install quantum-memory-graph
```

### Quick Start

```python
from quantum_memory_graph import store, recall

# Store memories
store("Project Alpha uses React frontend with TypeScript.")
store("Project Alpha backend is FastAPI with PostgreSQL.")

# Recall — graph traversal + QAOA finds the optimal combination
result = recall("What is Project Alpha's full tech stack?", K=4)
```

### Custom Configuration

```python
from quantum_memory_graph import MemoryGraph, recall
from quantum_memory_graph.pipeline import set_graph

graph = MemoryGraph(similarity_threshold=0.25)
set_graph(graph)

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

## Running the Official Benchmark

```bash
# Full LongMemEval 500-question benchmark
python3 benchmarks/run_longmemeval_chunked_staged.py --force
```

Requires `sentence-transformers`, `torch`, and the LongMemEval dataset (via Hugging Face).

## Requirements

- Python ≥ 3.9
- sentence-transformers
- networkx
- qiskit + qiskit-aer
- numpy

## IBM Quantum Hardware (Optional)

For production QAOA on real hardware:

```bash
pip install quantum-memory-graph[ibm]
export IBM_QUANTUM_TOKEN=your_token
```

Validated on IBM Quantum backends.

## Links

- GitHub: <https://github.com/Dustin-a11y/quantum-memory-graph>
- PyPI: `pip install quantum-memory-graph`
- LongMemEval: <https://arxiv.org/abs/2410.10813>
