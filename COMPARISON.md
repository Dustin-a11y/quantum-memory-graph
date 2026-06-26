# mem0 vs Quantum Memory Graph

mem0 is the most popular AI memory library, providing a simple API to store and retrieve memories for AI agents. Quantum Memory Graph (QMG) takes a fundamentally different approach — relationship-aware knowledge graphs with quantum-optimized subgraph selection instead of flat vector search.

## Feature Comparison Table

| Feature | mem0 | QMG |
|---------|------|-----|
| Architecture | Flat vector DB | Knowledge graph + embeddings |
| Memory relationships | No native support | Weighted edges (semantic, entity, temporal, source) |
| Retrieval | Cosine similarity only | Graph traversal + QAOA subgraph optimization |
| LongMemEval NDCG@10 | Not submitted | 94.26% (#1) |
| R@1 | — | 90.6% |
| Quantum optimization | No | Yes (QAOA, optional — works classically too) |
| API server | Yes | Yes |
| Python support | Yes | Yes |
| Obsidian integration | No | Yes (community plugin) |
| Conda install | No | Yes (conda-forge pending) |
| Docker | Yes | Yes (ghcr.io) |
| npm wrapper | No | Yes (qmg npm package) |
| License | Apache 2.0 | MIT |

## When to use mem0

- Simple "remember everything" use case — store and search memories with minimal setup
- Need battle-tested production stability with a large user base
- Flat semantic search is sufficient for your retrieval needs
- You don't need to understand *how* memories relate to each other

## When to use QMG

- Need **related memories returned together** — QMG's graph traversal retrieves connected clusters, not isolated documents
- Building **agent systems where context matters** — related facts (tech stack, team decisions, dependencies) are retrieved as a cohesive set
- Want **state-of-the-art retrieval performance** — #1 on LongMemEval with 94.26% NDCG@10
- **Obsidian or local-first workflows** — community plugin for personal knowledge management
- **MIT license** preferred over Apache 2.0

## Quick migration

```python
# mem0
m.add("message", user_id="user1")
results = m.search("query", user_id="user1")

# QMG equivalent
from quantum_memory_graph import store, recall
store("message")
results = recall("query", K=5)
```

## Using Both Together

mem0 and QMG can complement each other in the same application:

- Use **mem0** for simple, flat memory storage — quick facts, transient notes, user preferences that don't need relationship context
- Use **QMG** for relationship-aware retrieval — project decisions, technical dependencies, interconnected knowledge where the connections between facts matter as much as the facts themselves

This hybrid approach gives you mem0's simplicity for straightforward lookups plus QMG's graph intelligence for complex retrieval scenarios.
