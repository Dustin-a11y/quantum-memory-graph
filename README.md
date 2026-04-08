# Quantum Memory Graph ⚛️🧠

**Relationship-aware memory for AI agents. Knowledge graphs + quantum-optimized subgraph selection.**

Every memory system treats memories as independent documents — search, rank, stuff into context. But memories aren't independent. They have *relationships*. "The team chose React" becomes 10x more useful paired with "because of ecosystem maturity" and "FastAPI handles the backend."

Quantum Memory Graph maps these relationships, then uses QAOA to find the optimal *combination* of memories — not just the most relevant individuals, but the best connected subgraph that gives your agent maximum context.

## Benchmarks

### LongMemEval (ICLR 2025) — Industry Standard

500 questions across 53 conversation sessions. The gold standard for AI memory retrieval.

| System | R@5 | R@10 | NDCG@10 |
|--------|-----|------|---------|
| **Quantum Memory Graph (gte-large)** | **96.6%** | **98.7%** | **94.3%** |
| MemPalace raw | 96.6% | 98.2% | 88.9% |
| Quantum Memory Graph (e5-large) | 96.0% | 98.1% | 94.6% |
| Quantum Memory Graph (bge-large) | 95.9% | 98.2% | 94.0% |
| OMEGA | 95.4% | — | — |
| Mastra OM | 94.9% | — | — |
| Quantum Memory Graph (MiniLM, default) | 93.4% | 97.4% | 90.8% |

**#1 to our knowledge. Tied on R@5, best R@10 and NDCG@10 among published results.** Free. Open source.

Use `model="thenlper/gte-large"` for #1 accuracy, or `model="intfloat/e5-large-v2"` for best NDCG ranking.

### MemCombine — Combination Recall (250 Scenarios)

| Method | Coverage | Evidence Recall | F1 | Perfect |
|--------|----------|----------------|----|---------|
| Embedding Top-K | 92.3% | 93.9% | 91.3% | 181/250 |
| **Graph + QAOA** | **96.2%** | **97.7%** | **95.1%** | **212/250** |

Graph-aware quantum selection beats pure similarity by +3.8% on combination tasks.

## Choosing a Model

| Model | Size | GPU? | R@5 | Best For |
|-------|------|------|-----|----------|
| `all-MiniLM-L6-v2` (default) | 90MB | No | 93.4% | Laptops, CI/CD, quick prototyping |
| `BAAI/bge-large-en-v1.5` | 1.3GB | Recommended | 95.9% | Production servers with GPU |
| `intfloat/e5-large-v2` | 1.3GB | Recommended | 96.0% | Best ranking quality (NDCG) |
| `thenlper/gte-large` | 1.3GB | Recommended | 96.6% | Maximum retrieval accuracy |

```python
from quantum_memory_graph import MemoryGraph

# Default — works everywhere, no GPU needed
mg = MemoryGraph()

# High accuracy — needs ~2GB RAM, GPU speeds it up 60x
mg = MemoryGraph(model="thenlper/gte-large")
```

The default model runs on any machine. Larger models need more RAM and benefit from a GPU but aren't required — they'll just be slower on CPU.

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

## Deploying for AI Agents

### Replace Your Current Memory System

QMG is a drop-in upgrade for existing memory systems (Mem0, LangChain memory, custom RAG):

```python
# Before (typical flat similarity search)
results = memory.search("What's the tech stack?", k=5)

# After (graph-aware combination retrieval)
from quantum_memory_graph import store, recall

result = recall("What's the tech stack?", K=5)
# Returns connected memory clusters, not just individual matches
```

### Run as a Microservice

Deploy the API server for multiple agents to share:

```bash
pip install quantum-memory-graph[api]

# Default model (lightweight, no GPU)
python -m quantum_memory_graph.api --port 8502

# High accuracy (needs GPU for best speed)
QMG_MODEL=thenlper/gte-large python -m quantum_memory_graph.api --port 8502
```

Then from any agent:
```python
import requests

# Store a memory
requests.post("http://localhost:8502/store", json={"text": "User prefers dark mode"})

# Recall with graph + QAOA
result = requests.post("http://localhost:8502/recall", json={"query": "What are the user's preferences?", "K": 5})
```

### Migrate from Mem0 / LangChain

```python
from quantum_memory_graph import store

# Export your existing memories and bulk import
for memory in existing_memories:
    store(memory["text"], metadata=memory.get("metadata"))
# Graph connections are built automatically during import
```

### Production Tips

- **Shared API server**: Run one instance, point all agents at it. The knowledge graph is shared — Agent A's memories help Agent B's recall.
- **Model choice**: Use `gte-large` on GPU servers (96.6% accuracy). Use default `MiniLM` on laptops or CI (93.4%, no GPU needed).
- **Batch import**: Use `/store-batch` endpoint for bulk migration — 10x faster than individual stores.
- **Persistence**: Graph state saves to disk automatically. Restart the server without losing memories.

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
