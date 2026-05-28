# LongMemEval Official Benchmark Results — QMG v1.2 🏆

**Date:** May 28, 2026  
**Model:** thenlper/gte-large (1024-dim sentence transformer)  
**Pipeline:** Chunked two-stage (v7-style chunking + cosine retrieval)  
**Hardware:** DGX Spark GB10 (CUDA)  
**Total time:** 3158s (~53 min for 500 questions)

## Official Results

| Metric | Chunked gte-large | Plain (flat) gte-large | Chunking Delta |
|--------|:-----------------:|:----------------------:|:--------------:|
| R@1 | **90.6%** | 82.2% | **+8.4%** |
| R@5 | **98.6%** | 95.4% | **+3.2%** |
| R@10 | **99.4%** | 97.4% | **+2.0%** |
| NDCG@10 | **0.9426** | 0.8882 | **+0.0544** |

## Method

### Chunking Technique (500-char blocks, 100-char overlap)
1. Split each session into 500-character blocks with 100-character overlapping boundaries
2. Embed each chunk with gte-large (normalized)
3. Score question-chunk similarity via cosine similarity
4. Per-session score = mean of top-3 chunk cosine scores
5. Rank sessions by score

### Why chunking works
Session-level embedding dilutes signal because a session may contain 50+ turns spanning multiple topics. A 500-char chunk isolates the relevant sub-discussion, and mean-of-top-3 scoring ensures the session is ranked by its strongest evidence blocks rather than its average signal.

## Reproduction

```bash
# Prerequisites
pip install sentence-transformers torch numpy

# Data: download LongMemEval cleaned dataset
# https://github.com/... (LongMemEval official repo)

# Run full 500-question benchmark
python3 benchmarks/run_longmemeval_chunked_staged.py --force
```

## Comparison to Previous

| Version | Technique | R@5 | R@10 | When |
|---------|-----------|:---:|:----:|:----:|
| v1.0 | graph + QAOA (flat embed) | 85.7% | — | Apr 7 |
| v1.1 | v7 chunking pipeline | 95.8% | 98.85% | May 3 |
| **v1.2 (this)** | **Chunked gte-large + cosine** | **98.6%** | **99.4%** | **May 28** |

## Leaderboard Context

| Method | R@5 | R@10 | NDCG@10 |
|--------|:---:|:----:|:-------:|
| OMEGA (prev SOTA) | 89.2% | 94.1% | 87.5% |
| Mastra OM | 91.0% | 95.2% | 89.1% |
| QMG v1.1 (published #1) | 95.8% | 98.85% | 93.2% |
| **QMG v1.2 (this run)** | **98.6%** | **99.4%** | **0.9426** |

## Files

- `run_longmemeval_chunked_staged.py` — The official benchmark script
- `longmemeval_chunked_staged_results.json` — Full per-question results (500 entries)
- `longmemeval_chunked_staged_results.csv` — Tabular results
