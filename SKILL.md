---
name: quantum-memory-graph
description: >-
  Full memory system for AI agents — knowledge graphs + QAOA optimization +
  semantic tiers + deduplication + cross-agent sharing. #1 R@10 on LongMemEval
  (ICLR 2025). Use when: setting up agent memory, upgrading from flat vector
  search (Mem0/LangChain), running QMG queries, managing memory tiers,
  cross-agent sharing, dedup tuning, Obsidian vault export, or benchmarking
  memory quality. Includes quantum-researcher (Grover search, quantum walks)
  and quantum-agent-memory (QAOA compaction/recall). PyPI: quantum-memory-graph.
  API running on DGX Spark port 8501/8502.
---

# Quantum Memory Graph (QMG)

Full memory system for AI agents. Knowledge graphs + QAOA + tiers + dedup + sharing.
**#1 R@10 on LongMemEval** (98.7% R@10, 94.3% NDCG@10).

## Location

- **Main repo**: `/home/dt/Projects/quantum-memory-graph/`
- **GitHub**: `github.com/Dustin-a11y/quantum-memory-graph` (v1.0.0)
- **PyPI**: `pip install quantum-memory-graph`
- **API**: `http://100.124.61.124:8501` (primary), `:8502` (alt)
- **Mem0**: `http://100.124.61.124:8500`
- **Researcher**: `http://100.124.61.124:8505` (quantum-researcher API)

## Related Repos (all under QMG umbrella)

- `/home/dt/Projects/quantum-researcher/` — Grover search + quantum walks
- `/home/dt/Projects/quantum-agent-memory/` — QAOA compaction/recall layers
- `/home/dt/Projects/quantum-memory/` — Original QMG skill package

## Architecture

Three-tier memory: Hot (in-memory, <1hr) → Warm (SQLite, 1-24hr) → Cold (knowledge graph + QAOA)

Plus: dedup engine (cosine ≥0.95), cross-agent sharing, Obsidian vault export.

## Python Usage

```python
from quantum_memory_graph import store, recall

# Store — auto-builds knowledge graph + tier placement
store("Project Alpha uses React frontend with TypeScript.")
store("Project Alpha backend is FastAPI with PostgreSQL.")

# Recall — graph traversal + QAOA finds optimal combination
result = recall("What is Project Alpha's full tech stack?", K=4)
for memory in result["memories"]:
    print(f"  {memory['text']}")
```

## API Endpoints

```bash
# Store memory
curl -X POST http://100.124.61.124:8501/store \
  -H "Content-Type: application/json" \
  -d '{"text": "Important memory content", "metadata": {"agent": "bowser"}}'

# Recall
curl -X POST http://100.124.61.124:8501/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "What do I know about X?", "K": 5}'

# Batch store
curl -X POST http://100.124.61.124:8501/store-batch \
  -d '{"memories": [{"text": "mem1"}, {"text": "mem2"}]}'

# Stats
curl http://100.124.61.124:8501/stats
```

## Quantum Researcher (sub-module)

```bash
# Grover search — O(√N) document filtering
curl -X POST http://100.124.61.124:8505/search \
  -d '{"keywords": ["quantum"], "items": [{"text": "quantum paper"}, {"text": "cooking"}]}'

# Quantum walk — find hidden knowledge graph connections
curl -X POST http://100.124.61.124:8505/walk \
  -d '{"edges": [["A","B"],["B","C"]], "start": "A"}'
```

## Key Modules

- `graph.py` — Knowledge graph construction + traversal
- `subgraph_optimizer.py` — QAOA subgraph selection
- `tiers.py` — Hot/Warm/Cold tier management
- `dedup.py` — Cosine similarity deduplication
- `sharing.py` — Cross-agent memory sharing
- `recency.py` — Time-decay scoring
- `pipeline.py` — Full store/recall pipeline
- `api.py` — FastAPI REST server
- `obsidian.py` — Obsidian vault export with wikilinks

## Benchmarks

- **LongMemEval**: 98.7% R@10, 94.3% NDCG@10 (#1 published)
- **MemCombine**: 97.0% coverage, 98.6% evidence recall (tuned QAOA)
- **Optimal weights**: α=0.3, β=0.15, γ=0.35, threshold=0.25

## Install

```bash
pip install quantum-memory-graph                # basic
pip install quantum-memory-graph[api]            # with FastAPI server
pip install quantum-memory-graph[ibm]            # with IBM Quantum hardware
pip install "quantum-memory-graph[api]" quantum-researcher[all]  # everything
```

## OpenClaw Integration

QMG is wired into all agents via the `mem0-bridge` plugin. The plugin hooks into
`before_prompt_build` to inject relevant memories. Config in each agent's
`openclaw.json` under `plugins.entries.mem0-bridge`.

## Run Tests

```bash
cd /home/dt/Projects/quantum-memory-graph
pytest
```
