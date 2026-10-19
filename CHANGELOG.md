# Changelog

## 1.3.0 (2026-10-19)
- Bring the deployed L3 safety layer under version control. Seven modules that
  existed only in the release working tree are now in the repo:
  `rollback`, `semantic`, `admission`, `inventory_v2`, `cache_attestation`,
  `learning`, `search`. Without this, the code that hash-pins and validates
  every production graph pair had no source of record and no off-machine copy.
- Sync `api`, `graph`, `pipeline`, `recency` and `__init__` with the deployed
  release, so the repo matches what actually runs and is graded by CI.
- Deliberately NOT synced: `pce_optimizer` and `subgraph_optimizer`. The deployed
  versions diverge from this repo's pinned tests in two ways and need a
  deliberate decision rather than a silent copy:
  1. `pce_encode_adjacency` in the deployed version adds `weight * 0.5` once per
     shared qubit instead of once per (qi, qj) pair, inflating shared-qubit
     weights by the number of shared qubits. That is exactly the double count
     `tests/test_pce_optimizer.py` exists to prevent.
  2. `optimize_subgraph` in the deployed version returns method `"greedy"`
     where the repo's CodeQL-cleanup test pins `"greedy_fallback"`.
  Both are open findings, not repo regressions; the repo keeps the version its
  tests pin until the deployed behaviour is reviewed.
- Fix: `MemoryGraph.load()` now strips third-generation attributes that no
  current code reads (`sources`, `_source_index`, `max_connections`) under
  guard. Pairs written by the older package previously failed the strict pair
  validator forever: the loader added `semantic_store` but never removed the
  legacy fields, so `save()` re-emitted a pair its own validator rejected. The
  strip refuses to drop anything that carries data — non-empty `sources` or
  `_source_index`, or a non-int `max_connections`, raise instead.
  Tested in `tests/test_legacy_attr_migration.py` (7 tests), including the
  round-trip regression that an old pair becomes validator-clean after one
  load/save cycle.
- Security: fix quadratic regex backtracking (ReDoS) in
  `MemoryGraph._extract_temporal_entities`. Three unbounded quantifiers
  (`\w+`, `\d+`) sat directly ahead of a required `\s+`, so a memory text of one
  long token backtracked quadratically: 20k digits = 6-9s, 50k digits = ~115s of
  blocked extraction. `add_memory` calls this, and `POST /memory` reaches
  `add_memory`, so any API caller could stall a service for minutes. Quantifiers
  are now length-capped, which bounds the engine to a constant number of splits
  per position. 50k digits: 115s -> 0.12s. Output on 2,000 real memories is
  byte-identical, and `tests/test_temporal_extraction_redos.py` pins both the
  timing and the unchanged extraction results.
- Note: units running with a `QMG_MODEL` override still fail their startup pair
  check on an untouched legacy pair, because the validator inspects the raw
  pickle before `load()` can strip anything. Those pairs need an explicit
  cold-bootstrap conversion.

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
