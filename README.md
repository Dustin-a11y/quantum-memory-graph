# Quantum Memory Graph ⚛️🧠

**Relationship-aware memory for AI agents. Knowledge graphs + quantum-optimized subgraph selection.**

Every memory system treats memories as independent documents — search, rank, stuff into context. But memories aren't independent. They have *relationships*. "The team chose React" becomes 10x more useful paired with "because of ecosystem maturity" and "FastAPI handles the backend."

Quantum Memory Graph maps these relationships, then uses QAOA to find the optimal *combination* of memories — not just the most relevant individuals, but the best connected subgraph that gives your agent maximum context.

## Benchmark: MemCombine

We created MemCombine to test what no existing benchmark measures — **memory combination quality**.

| Method | Coverage | Evidence Recall | F1 | Perfect |
|--------|----------|----------------|----|---------|
| Embedding Top-K | 69.9% | 65.6% | 68.1% | 1/5 |
| **Graph + QAOA** | **96.7%** | **91.0%** | **92.6%** | **4/5** |
| **Advantage** | **+26.8%** | **+25.4%** | **+24.5%** | |

When the task is "find memories that work *together*," graph-aware quantum selection crushes pure similarity search.

## Install

```bash
pip install quantum-memory-graph
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
│  1. Graph Search     │  Embedding similarity + multi-hop traversal
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
   - Entity co-occurrence (shared people, projects, concepts)
   - Temporal proximity (memories close in time)
   - Source proximity (same conversation/document)

2. **Subgraph Optimizer** (`subgraph_optimizer.py`) — QAOA circuit that maximizes:
   - α × relevance (individual memory scores)
   - β × connectivity (edge weights within selected subgraph)
   - γ × coverage (topic diversity across selection)

3. **Pipeline** (`pipeline.py`) — Unified `store()` and `recall()` interface.

### Optional: MemPalace Integration

Use [MemPalace](https://github.com/milla-jovovich/mempalace) (MIT, by @bensig) as the storage/retrieval backend for 96.6% base retrieval quality:

```python
from quantum_memory_graph.mempalace_bridge import store_memory, recall_memories

# MemPalace stores verbatim → ChromaDB retrieves candidates → QAOA selects optimal subgraph
result = recall_memories("What happened in the meeting?", K=5, use_qaoa=True)
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

### Run MemCombine Benchmark

```python
from benchmarks.memcombine import run_benchmark

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

Built with [MemPalace](https://github.com/milla-jovovich/mempalace) by @bensig (MIT License). See [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES).

## Links

- [quantum-agent-memory](https://github.com/Dustin-a11y/quantum-agent-memory) — The QAOA optimization engine
- [MemPalace](https://github.com/milla-jovovich/mempalace) — Storage and retrieval backend
- [MemCombine Benchmark](benchmarks/memcombine.py) — Test memory combination quality
