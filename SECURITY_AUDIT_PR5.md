# PR #5 Security & Claim Audit Report

**Repo:** `Dustin-a11y/quantum-memory-graph`
**PR:** #5 — "Bench'd champion proof — 86.2% trace accuracy (431/500)"
**Branch:** `benchd-champion-862` → `main`
**Date:** 2026-07-18
**Auditor:** Independent subagent review
**Scope:** Security, privacy, licensing, and claim correctness. Read-only. No modifications.

---

## Executive Summary

| Category | Verdict | Count |
|----------|---------|-------|
| BLOCKER | QAOA claim unsupported by signed evidence | 1 |
| WARNING | Coverage gaps, disclosure gaps | 4 |
| OK | Verified correct | 11 |

**Overall: DO NOT MERGE without fixing the BLOCKER.** The signed manifest contains zero evidence for the "3/500 QAOA-eligible" claim repeated across PR body, README, PROVENANCE.md, and commit message. The validator silently reports 0/500 with no error.

---

## BLOCKER

### B1. QAOA eligibility claim (3/500) is absent from signed manifest

**Severity:** BLOCKER
**Claim:** "QAOA eligible: 3/500 (0.6%) — classical-dominant" appears in:
- PR body (`## Bench'd Harness Champion Proof`)
- README.md table
- PROVENANCE.md header and disclosure section  
- Commit message body

**Evidence in manifest:** ZERO traces show QAOA eligibility.

| Check path | Result |
|-----------|--------|
| `traces[].metadata.qaoa_eligible == True` | 0/500 — no trace has a `metadata` field at all |
| `traces[].scoring_method == "qaoa"` | 0/500 — all 500 traces use `scoring_method: "llm"` |
| `manifest.scores.qaoa_eligible_count` | Field does not exist |
| `"qaoa"` string anywhere in trace raw_recall | 0/500 |
| `"qaoa"` string anywhere in trace response | 0/500 |

**The 3 traces that match "quantum" in raw_recall** (`trace_e922a6f56e9b`, `trace_5ba22ee56b4d`, `trace_d6f7553c774b`) contain benchmark synthetic conversations about quantum physics, the Orch-OR theory of consciousness, and camera sensor quantum efficiency — **not QAOA graph optimization.** The word "quantum" appears in conversation content, not in retrieval telemetry.

**Validator behavior:** The validator's `qaoa_eligibility` check (validator.py:169-182) tries three paths, all return 0, and reports `"QAOA eligible: 0/500 (0.0%)"` — silently contradicting the published claim. The corresponding test `test_qaoa_eligible_is_small_fraction` passes trivially (`assert 0 <= 5.0`), masking the discrepancy.

**Root cause possibilities:**
1. The QAOA telemetry was not embedded in the manifest at signing time (pipeline bug — `metadata` field missing on all traces)
2. The 3/500 number was counted from server-side logs/QMG telemetry but not captured in the signed artifact
3. The number 3 is wrong

**Recommended action:** Either (a) re-sign the manifest with QAOA eligibility metadata properly embedded in trace-level `metadata` fields, or (b) remove the "3/500" claim from all documentation and state that QAOA telemetry was not captured in this run. Option (b) is acceptable since the claim of "classical-dominant" is independently true (all 500 traces use `scoring_method: "llm"`).

---

## WARNINGS

### W1. Secret scanning does not cover the 30MB manifest

**Severity:** WARNING
**Location:** `tests/test_validator.py:170-195`

The `test_no_api_keys_in_summary` test scans only `summary.json` (2 KB). The `test_no_home_paths_in_artifacts` test scans only `summary.json`, `validator.py`, and `PROVENANCE.md`. The 30 MB `manifest.signed.json` with 500 full conversation traces is **never scanned by any test.**

I performed a manual scan of the full manifest — no API keys, tokens, or local paths found. The existing tests are functionally correct for the current data but structurally incomplete for future runs.

**Recommended action:** Add a test scanning the full manifest, or at minimum document that the full manifest was manually verified for this publication.

### W2. LongMemEval full benchmark redistribution

**Severity:** WARNING

The `manifest.signed.json` contains the complete LongMemEval benchmark dataset inside 500 traces:
- All question texts, expected answers, question IDs
- Full conversation histories (ingest_history with timestamps)
- Judge reasoning (LLM judge evaluation text)
- Generated answers and scoring

LongMemEval is MIT-licensed, so redistribution is **legally permitted**. However, publishing expected answers, judge reasoning, and question IDs alongside scored traces gives future test-takers access to the ground truth. This compromises benchmark integrity for anyone who discovers this repository before running their own evaluation.

**Recommended action:** Add a disclosure in PROVENANCE.md noting that the manifest contains full benchmark data including expected answers.

### W3. Validator fails silently on QAOA claim mismatch

**Severity:** WARNING
**Location:** `validator.py:169-182`

The validator reports `"QAOA eligible: 0/500 (0.0%)"` with no error, no warning, and no comparison against the expected value of 3. The `EXPECTED_*` constants at the top of the file (lines 27-43) don't include an expected QAOA count. The test `test_qaoa_eligible_is_small_fraction` only checks `pct <= 5.0`, which passes for any value ≤ 25.

**Recommended action:** Add `EXPECTED_QAOA_ELIGIBLE = 3` to the validator constants and verify it in the validation output. This turns a silent pass into an explicit check.

### W4. No CI workflow for automated validation

**Severity:** WARNING

The PR adds a self-contained validator with 18 tests but no GitHub Actions workflow. Without CI, the validator only runs when someone manually executes it. Future modifications to the manifest or validator could break verification silently.

**Recommended action:** Add a minimal GitHub Actions workflow that runs `python validator.py && pytest tests/ -v` on push to the benchmark directory.

---

## OK (Verified)

### O1. Ed25519 canonicalization — ✅

- Manifest hash: SHA-256 of `json.dumps(manifest, sort_keys=True, separators=(",", ":"))`  
- Signature: Ed25519 over the **hex string** of the manifest hash (NOT raw bytes)
- Verifying with `cryptography` library passes
- Canonicalization is deterministic (verified via double-serialization test)
- Public key: `483b133c089ac3ff5fbb3c6df75923ac3874d550e3997ff6c617572a5aa5a830`
- Fingerprint: `23442739ed37e98f`

### O2. File integrity — ✅

- File SHA-256 matches expected (`01a1b981e4652598976b7f32ac9e5a4f5261df3585b72c27977ea4a26a29be82`)
- Manifest hash matches expected (`c8e6d9cb4026f017e421c56be6cabda1a1d4958f0a4def2557b4150fdd02ef16`)
- Manifest hash matches stored value in signed JSON

### O3. Score integrity — ✅

- Trace accuracy: 431 passed / 69 failed / 500 total = 86.2% (verified from trace-level `scored_correct`)
- Nuance overall: 85.66936237178193 → rounds to 85.67
- Nuance dimensions: recall 87.18, temporal 88.63, reasoning 81.20
- Score summary embedded in manifest matches summary.json

### O4. All 18 tests pass — ✅

`pytest tests/ -v` — 18/18 passed in 1.46s.

### O5. Self-run / not-leaderboard disclosure — ✅

Clearly disclosed in: PR title, PR body, README, PROVENANCE.md disclosure section. No ambiguity.

### O6. Score disambiguation (86.2 vs 85.67) — ✅

PR, README, PROVENANCE.md, and summary.json all correctly distinguish:
- **Trace accuracy (86.2%):** Pass/fail count from CLI trace
- **Nuance overall (85.67):** Weighted metric from `scores.nuance`

The summary.json even includes an explicit `_disclaimer` field: "nuance scores are distinct from trace accuracy."

### O7. Classical-dominant claim direction — ✅

All 500 traces use `scoring_method: "llm"` (classical subgraph selection). The run is genuinely classical-dominant regardless of the QAOA count dispute.

### O8. No credentials or API keys — ✅

Manual scan of full 30MB manifest + automated scan of text files: no API keys, tokens, Bearer auth strings, or credential patterns found.

### O9. No local paths leaked — ✅

No `/home/dt` paths in any artifact. Internal IPs found (192.168.x.x, 10.0.0.1) are in synthetic benchmark conversation context, not real infrastructure.

### O10. GitHub size — ✅

30MB single file, 113,888 line additions. Under GitHub's 100MB limit. Acceptable for a proof publication.

### O11. PII assessment — ✅ (acceptable)

All conversation data in traces is synthetic LongMemEval benchmark content. Phone numbers are synthetic (555-123-4567) or toll-free helpline references (800-273-8255 National Suicide Prevention Lifeline — in benchmark context about mental health resources). Email domains are synthetic (example.com, cityviewrooftop.com, etc.). Addresses are synthetic (123 Main St, 6801 Hollywood Blvd). No real personal data found.

---

## Detailed Findings

### Changed files in PR

| File | Additions | Risk |
|------|-----------|------|
| `README.md` | +26, -1 | Claims about QAOA (see B1) |
| `PROVENANCE.md` | +124 | Claims about QAOA (see B1) |
| `manifest.signed.json` | +113,222 (30MB) | Full benchmark data (see W2) |
| `summary.json` | +74 | Lightweight scores — clean |
| `validator.py` | +234 | Silent QAOA check (see W3) |
| `tests/test_validator.py` | +208 | Coverage gaps (see W1) |
| `tests/__init__.py` | 0 | Empty file |

### Validator execution evidence

```
Validator — Bench'd manifest: .../manifest.signed.json
  File exists:         ✅
  File SHA-256:        ✅  01a1b981e4652598...
  Manifest hash:       ✅  c8e6d9cb4026f017...
  Ed25519 signature:   ✅  verified
  Key fingerprint:     ✅  23442739ed37e98f
  Score summary:       ✅  431/500
  Trace accuracy:      ✅  431/500 = 86.2%
  QAOA eligible:       0/500 (0.0%)    ← contradicts 3/500 claim
  OVERALL:             ✅
```

### Manifest trace structure

Each trace (no `metadata` field present):
- `trace_id`, `question_id`, `dimension`, `status`, `scored_correct`
- `scoring_method`: always `"llm"` (never `"qaoa"`)
- `ingest_history`: full multi-turn conversation (synthetic)
- `expected_answer`, `generated_answer`, `judge_reasoning`
- `latency_ms`, `recall_tokens`, `ingest_tokens`
- `retrieval_hit`, `partial_hit`, `score`, `max_score`
- `grounded`, `hallucination_risk`, `abstained`
- `word_overlap`, `answer_density`, `compression_ratio`

No `metadata` field on any of the 500 traces.

---

## Recommendations (Prioritized)

1. **[BLOCKER — Fix before merge]** Resolve the QAOA 3/500 discrepancy:
   - Option A: Re-sign manifest with QAOA telemetry in trace `metadata` fields
   - Option B: Remove "3/500" claims, state QAOA telemetry not captured, keep classical-dominant claim (which is independently true)

2. **[WARNING]** Add expected QAOA count to validator constants and verify it explicitly

3. **[WARNING]** Add full-manifest secret scanning test or document manual verification

4. **[WARNING]** Add disclosure about full benchmark data redistribution in PROVENANCE.md

5. **[WARNING]** Add minimal CI workflow to run validator + tests automatically
