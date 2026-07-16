# Native LongMemEval Official Run — 2026-07-16

## Overview

Self-run evaluation of QMG v1.3 (chunked BM25 hybrid retrieval) on the official [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark (ICLR 2025), using the exact official repository and evaluation pipeline.

**Overall end-to-end QA accuracy: 85.6% (428/500)**

## Artifacts

| File | SHA-256 | Description |
|------|---------|-------------|
| `results/hypotheses_merged_500.jsonl` | `a4c5aea53fa484055120c89d5e06dfe35cbfa43c125417c668c62d955c86420d` | 500 merged hypotheses (486 original + 14 repaired) |
| `results/eval-results-500.json` | `7b79e44baf53cbf8b139fa9a7e29df1347209910677cf69d8888c8b74210b6a3` | Per-item evaluation results (judge outputs, token usage, cost) |
| `results/eval-summary.json` | `28171bd89fe58afa1ee75ea64cb1c87ec380b2b7666765fb2a385883495fbea2` | Aggregate evaluation summary |

## Methodology

### Retrieval
- **Method:** `qmg-bm25-hybrid-70-30-chunked-session-v1.3`
- **Repository:** Official `xiaowu0162/LongMemEval` at commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`
- **Dataset:** Official cleaned dataset, SHA `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- **Items:** 500/500 completed, 0 errors
- **Recall metrics (recall_any):** R@1 88.1%, R@5 96.8%, R@10 98.7%
- **Recall metrics (recall_all, official strict parser):** recall_all@5 0.8723, ndcg_any@5 0.8993, recall_all@10 0.9511, ndcg_any@10 0.9176
- **Runner:** `scripts/qmg_chunked_hybrid_runner.py`

### Generation
- **Model:** DeepSeek R1 via OpenRouter
- **Top-K:** 10 retrieved sessions
- **Prompt:** Official `prepare_prompt` from LongMemEval repository
- **Items:** 500 hypotheses (486 in first pass + 14 repaired)
- **Scripts:** `scripts/run_generation_openrouter.py`, `scripts/run_generation_repair.py`

### Evaluation
- **Judge:** GPT-4o via OpenRouter
- **Prompt:** Exact `get_anscheck_prompt` from official LongMemEval evaluator
- **Script:** `scripts/evaluate_qa_openrouter.py`
- **Cost:** $0.547515 (214,726 input + 1,070 output tokens)

## Results

### Overall
| Metric | Value |
|--------|-------|
| Total items | 500 |
| Correct | 428 |
| Incorrect | 72 |
| Errors | 0 |
| **Accuracy** | **85.6%** |

### By Question Type
| Type | Accuracy | Count |
|------|:--------:|:-----:|
| single-session-user | 98.57% | 70 |
| single-session-assistant | 96.43% | 56 |
| knowledge-update | 87.18% | 78 |
| temporal-reasoning | 81.95% | 133 |
| single-session-preference | 80.00% | 30 |
| multi-session | 78.20% | 133 |

## Disclosure

- **Self-run:** This evaluation was run independently on the official LongMemEval cleaned dataset. It has NOT been authored, verified, or endorsed by the LongMemEval maintainers.
- **Separate from Bench'd:** The Bench'd.ai score of 86.2 is a separate evaluation using a different harness. This artifact is the native official run only.
- **Credentials:** No API keys, tokens, or secrets are included in any published artifact. All scripts load credentials from runtime files (not committed).
- **Secret scan:** All published files were scanned with high-confidence regex patterns for API keys, tokens, private keys, and credential paths. No secrets found. Paths referencing `/home/dt` were sanitized to generic relative paths.

## Reproducing

All commands are run from this directory (`benchmarks/longmemeval/native-official-20260716`).

0. Install dependencies:
   ```bash
   pip install sentence-transformers torch numpy rank_bm25 openai tiktoken backoff
   ```

1. Clone the official LongMemEval repo:
   ```bash
   git clone https://github.com/xiaowu0162/LongMemEval.git
   cd LongMemEval && git checkout 9e0b455f4ef0e2ab8f2e582289761153549043fc
   cd ..
   ```

2. Download the official cleaned LongMemEval dataset:
   ```bash
   mkdir -p LongMemEval/data
   curl -L -o LongMemEval/data/longmemeval_s_cleaned.json \
     https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
   ```

   Verify SHA-256:
   ```bash
   echo "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442  LongMemEval/data/longmemeval_s_cleaned.json" | sha256sum -c
   ```

3. Create credential file (see `benchd-openrouter.env.example` for template):
   ```bash
   cp benchd-openrouter.env.example benchd-openrouter.env
   # Edit benchd-openrouter.env with your OpenRouter API key
   ```

   **Note on credential mechanism:** All scripts (`run_generation_openrouter.py`, `run_generation_repair.py`, `evaluate_qa_openrouter.py`) accept `--cred-file` as a CLI argument (default: `./benchd-openrouter.env`).

4. Run retrieval (reproduces QMG v1.3 published retrieval method):
   ```bash
   python scripts/qmg_chunked_hybrid_runner.py \
     --in-file LongMemEval/data/longmemeval_s_cleaned.json \
     --out-file results/retrieval.jsonl
   ```

   The runner is a self-contained script that does not import `quantum_memory_graph`; it reimplements the published QMG v1.3 chunked BM25 hybrid algorithm using sentence-transformers, numpy, and rank_bm25 for a clean-room reproduction path.

5. Run generation (requires OpenRouter API key via `--cred-file`):
   ```bash
   python scripts/run_generation_openrouter.py \
     --in-file results/retrieval.jsonl \
     --out-dir results/ \
     --cred-file ./benchd-openrouter.env
   ```

6. Run evaluation (requires OpenRouter API key; uses `--cred-file` for credential path):
   ```bash
   python scripts/evaluate_qa_openrouter.py \
     --hyp-file results/hypotheses_merged_500.jsonl \
     --ref-file LongMemEval/data/longmemeval_s_cleaned.json \
     --out-dir results/ \
     --cred-file ./benchd-openrouter.env
   ```

   Dry-run to validate setup without API calls:
   ```bash
   python scripts/evaluate_qa_openrouter.py \
     --hyp-file results/hypotheses_merged_500.jsonl \
     --ref-file LongMemEval/data/longmemeval_s_cleaned.json \
     --dry-run \
     --cred-file ./benchd-openrouter.env
   ```

## Tests

All scripts have corresponding test files in `tests/`:
- `tests/test_chunked_hybrid.py` — Retrieval pipeline tests
- `tests/test_generation_wrapper.py` — Generation wrapper tests
- `tests/test_generation_wrapper_repair.py` — Repair workflow tests
- `tests/test_evaluate_qa_openrouter.py` — Evaluation wrapper tests

Run with: `pytest tests/ -v`