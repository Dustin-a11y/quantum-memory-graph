# Changelog

## 1.2.1 (2026-05-31)
- Add synergy reranker: word-overlap synergy + diversity selection
- Wire synergy reranker into pipeline: recall(query, method="synergy")
- Remove MemCombine from repo
- Add link to LongMemEval submission issue #46

## 1.2.0 (2026-05-28)
- Official LongMemEval #1 renewed — R@5 98.6% (+2.8% over v1.1)
- Chunked retrieval with QAOA-CVaR subgraph refinement
- Switch default model to gte-large (96.6% R@5)

## 1.1.1 (2026-05-18)
- #1 on LongMemEval (R@5 95.8%, R@10 98.85%)
- PCE optimizer for warm-start accelerated QAOA
- IBM Quantum backend support (ibm_fez, kingston, marrakesh)
- Short-term memory layer with recency boosting
- Tiered retrieval: working memory → graph → QAOA

## 1.1.0 (2026-05-10)
- QAOA subgraph optimization with CVaR
- Multi-edge relationship types (semantic, temporal, entity, causal)
- FastAPI REST server with token auth
- Graph persistence (pickle save/load)

## 1.0.0 (2026-05-04)
- Initial release
- Knowledge graph memory with embedding-based retrieval
- Cosine similarity neighborhood search
- Entity extraction via spaCy
- sentence-transformers embedding support

## 0.1.0 (2026-04-07)
- First PyPI release
- Basic store/recall pipeline
- NetworkX graph backend
