# Bench'd Champion Proof — Run e375a77f15fb

## Overview

Self-run evaluation of QMG v1.4.0 on the Bench'd harness against the LongMemEval benchmark (ICLR 2025). **This is a self-run Bench'd harness artifact, not official leaderboard verification and not quantum-derived.**

**Trace accuracy: 86.2% (431/500)** | Nuance score: 85.67 | Classical pipeline (see provenance note)

## ⚠️ Score Distinction

This run produced two different scores that measure different things. **Do not conflate them.**

| Score | Value | What It Measures |
|-------|:-----:|------------------|
| **Trace accuracy** | **86.2%** (431/500) | Pass/fail count from the Bench'd harness CLI trace — raw correct answer count / total questions |
| **Nuance overall** | **85.67** | Bench'd internal weighted scoring across recall, temporal, and reasoning dimensions |

The trace accuracy (86.2%) is the headline number reported by the Bench'd harness. The nuance score (85.67) is a separately computed weighted metric from the manifest's `scores.nuance` field. Both are valid metrics; neither should be mislabeled as the other.

## Provenance

| Attribute | Value |
|-----------|-------|
| **Run ID** | `run_e375a77f15fb` |
| **System** | QMG v1.4.0 (quantum-memory-graph) |
| **Benchmark** | LongMemEval v1.0 (ICLR 2025) |
| **Harness** | Bench'd v0.1.0 |
| **Answerer model** | deepseek/deepseek-r1 |
| **Judge model** | openai/gpt-4o |
| **Judge temperature** | 0.0 |
| **Started** | 2026-07-14T23:33:14 UTC |
| **Completed** | 2026-07-15T03:10:40 UTC |
| **Duration** | ~3h 37m |
| **Mean latency** | 9,642.62 ms per question |

## Results

### Overall

| Metric | Value |
|--------|-------|
| Total questions | 500 |
| Passed | 431 |
| Failed | 69 |
| Pending | 0 |
| **Trace accuracy** | **86.2%** |

### Nuance Scores (weighted, distinct from trace accuracy)

| Dimension | Score |
|-----------|:-----:|
| Recall | 87.18 |
| Temporal | 88.63 |
| Reasoning | 81.20 |
| **Overall** | **85.67** |

### Retrieval

| Metric | Value |
|--------|:-----:|
| Recall@K (K=5) | 47.4% |
| Partial recall | 68.4% |
| Total hits | 237 |
| Partial hits | 342 |

### Faithfulness

| Metric | Value |
|--------|:-----:|
| Grounded | 32.6% (163/500) |
| Abstention | 9.8% (49/500) |
| Mean hallucination risk | 0.4965 |

## Cryptographic Verification

The manifest is cryptographically signed with Ed25519. The signing scheme signs the canonical SHA-256 manifest hash (not the full manifest JSON), allowing verification without loading the 30 MB file:

| Field | Value |
|-------|-------|
| **Canonical manifest hash** (SHA-256) | `c8e6d9cb4026f017e421c56be6cabda1a1d4958f0a4def2557b4150fdd02ef16` |
| **Ed25519 signature** | `6ebbde1ac3b8a38e7be328bbe31e7b499433d805...` |
| **Ed25519 public key** | `483b133c089ac3ff5fbb3c6df75923ac3874d550e3997ff6c617572a5aa5a830` |
| **Key fingerprint** | `23442739ed37e98f` |
| **Signed at** | 2026-07-15T03:10:40 UTC |
| **Signing mode** | local |
| **File SHA-256** | `01a1b981e4652598976b7f32ac9e5a4f5261df3585b72c27977ea4a26a29be82` |

## Provenance Note: Quantum Contribution

**The signed manifest contains no optimizer execution telemetry** — no trace-level `metadata.qaoa_eligible` field, no `scoring_method: "qaoa"` entries, and no manifest-level QAOA count. All 500 traces use `scoring_method: "llm"` (classical subgraph selection). The manifest therefore **cannot establish any quantum contribution** to these scores.

A separate code-path audit confirms the QMG pipeline includes QAOA subgraph optimization at the framework level, but whether any optimizer was invoked during this specific run is outside what the signed proof can verify. The published claim of "classical-dominant" is consistent with the manifest contents (all 500 traces are LLM-scored).

**Do not cite any specific QAOA-eligible count for this run** — no count is attested by the signed manifest. The manifest proves scores and signature integrity only; it does not contain optimizer execution telemetry.

## Disclosure

- **Self-run:** This evaluation was run independently using the Bench'd harness CLI. It has NOT been authored, verified, or endorsed by the Bench'd.ai maintainers and does NOT appear on any official leaderboard.
- **Not quantum-derived:** The 86.2% result is achieved through the classical retrieval pipeline (BM25 hybrid + chunked embeddings + subgraph selection).
- **Separate from native official run:** The `native-official-20260716/` directory contains a separate evaluation (85.6%) using the official LongMemEval repository directly. This Bench'd run uses the Bench'd harness, which wraps the benchmark with its own adapter, judge pipeline, and scoring logic.
- **Credentials:** No API keys, tokens, or secrets are included in any published artifact. The signed manifest contains only benchmark data, traces, and cryptographic signatures.

## Manifest Data & Reproducibility

The `manifest.signed.json` (30 MB) contains the full benchmark dataset from the LongMemEval benchmark (MIT-licensed), including:

- All 500 question texts and expected answers
- Full synthetic conversation histories (ingest_history with timestamps)
- Generated answers and LLM judge reasoning
- Per-trace scoring, retrieval, and faithfulness metrics

This data is published for **reproducibility** under the applicable LongMemEval dataset terms (MIT license). **Do not use the expected answers or judge reasoning labels for model tuning** — doing so would compromise benchmark integrity for any future evaluation.

## Artifacts

| File | Description |
|------|-------------|
| `manifest.signed.json` | Full signed Bench'd manifest (30 MB) — 500 traces, scores, cryptographic signature |
| `summary.json` | Lightweight summary — scores, hash, signature, without traces |
| `PROVENANCE.md` | This file — full provenance and disclosure |
| `validator.py` | Deterministic validator — verifies manifest hash and Ed25519 signature |
| `tests/test_validator.py` | Automated tests for the validator |

## Verifying

```bash
# Verify the manifest hash and Ed25519 signature
python validator.py

# Run automated tests
pytest tests/ -v
```

The validator is self-contained with no dependencies beyond Python 3.9+ stdlib (hashlib, json). For Ed25519 signature verification, it requires `cryptography` (pip install cryptography). Without it, the validator still verifies the manifest hash integrity.

### What the validator checks

1. **File SHA-256** — The `manifest.signed.json` file matches the expected hash
2. **Manifest hash** — The canonical JSON representation of the manifest matches the stored hash
3. **Ed25519 signature** — The signature over the manifest hash verifies against the public key
4. **Score integrity** — The embedded scores match the expected published values
5. **Trace accuracy** — 431/500 = 86.2% confirmed from trace data
