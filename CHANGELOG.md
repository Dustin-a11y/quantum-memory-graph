# Changelog

All notable changes to Quantum Memory Graph.

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
