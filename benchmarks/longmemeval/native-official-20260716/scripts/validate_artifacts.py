#!/usr/bin/env python3
"""
Artifact integrity validator for native LongMemEval published results.

Validates:
  1. JSON/JSONL parseability and item counts
  2. eval-summary.json consistency recomputed from eval-results-500.json
  3. SHA256 integrity against PROVENANCE.md (fail-closed: missing PROVENANCE
     or zero parsed hashes is a hard failure)
  4. Schema field presence in all records

All validation functions are pure: they accept paths/objects as arguments and
return lists of failure strings.  The main() entry point collects failures and
exits non-zero when any exist.  This keeps the validator fully testable without
global-state leakage.

Usage:
    python scripts/validate_artifacts.py
    python scripts/validate_artifacts.py --base-dir /path/to/artifacts   # override root

Exit 0 = all checks pass, non-zero = integrity failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── file paths (relative to a configurable base) ──────────────────────

DEFAULT_BASE = Path(__file__).resolve().parent.parent

RESULTS_DIR_NAME = "results"
PROVENANCE_NAME = "PROVENANCE.md"
EVAL_RESULTS_NAME = "eval-results-500.json"
HYPOTHESES_NAME = "hypotheses_merged_500.jsonl"
EVAL_SUMMARY_NAME = "eval-summary.json"


def resolve_paths(base: Path) -> Dict[str, Path]:
    """Return the canonical artifact paths rooted at *base*."""
    results = base / RESULTS_DIR_NAME
    return {
        "results_dir": results,
        "provenance": base / PROVENANCE_NAME,
        "eval_results": results / EVAL_RESULTS_NAME,
        "hypotheses": results / HYPOTHESES_NAME,
        "eval_summary": results / EVAL_SUMMARY_NAME,
    }


# ── helpers ───────────────────────────────────────────────────────────

def file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _print_failures(failures: List[str]) -> None:
    for msg in failures:
        print(f"FAIL: {msg}")


# ── 1. Parseability and counts ────────────────────────────────────────

def validate_parseability_and_counts(
    hypotheses_path: Path,
    eval_results_path: Path,
    eval_summary_path: Path,
) -> Tuple[List[str], Optional[list], Optional[dict]]:
    """Validate JSON/JSONL parseability, expected counts, and required fields.

    Returns (failures, eval_results_list, eval_summary_dict).
    """
    failures: List[str] = []

    # --- hypotheses ---
    if not hypotheses_path.exists():
        failures.append(f"Missing {hypotheses_path}")
        return failures, None, None
    hypotheses: list = []
    with open(hypotheses_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                hypotheses.append(json.loads(line))
            except json.JSONDecodeError as e:
                failures.append(
                    f"hypotheses_merged_500.jsonl line {i}: invalid JSON: {e}"
                )
    print(f"  hypotheses_merged_500.jsonl: {len(hypotheses)} entries")
    if not any("invalid JSON" in f for f in failures):
        print("    OK: JSONL parseable")
    if len(hypotheses) != 500:
        failures.append(
            f"Expected 500 hypotheses, got {len(hypotheses)}"
        )
    else:
        print(f"    OK: Count {len(hypotheses)}")
    for i, h in enumerate(hypotheses):
        for field in ("question_id", "hypothesis"):
            if field not in h:
                failures.append(f"hypothesis {i} missing field '{field}'")
    if not any("missing field" in f for f in failures):
        print("    OK: All hypotheses have required fields")

    # --- eval-results ---
    if not eval_results_path.exists():
        failures.append(f"Missing {eval_results_path}")
        return failures, None, None
    try:
        with open(eval_results_path) as f:
            eval_results: list = json.load(f)
    except json.JSONDecodeError as e:
        failures.append(f"eval-results-500.json: invalid JSON: {e}")
        return failures, None, None
    if not isinstance(eval_results, list):
        failures.append("eval-results-500.json is not a JSON array")
        return failures, eval_results, None
    print(f"  eval-results-500.json: {len(eval_results)} entries")
    if not any("not a JSON array" in f for f in failures):
        print("    OK: JSON parseable")
    if len(eval_results) != 500:
        failures.append(
            f"Expected 500 eval results, got {len(eval_results)}"
        )
    else:
        print(f"    OK: Count {len(eval_results)}")
    required_fields = [
        "question_id", "question_type", "question", "answer",
        "hypothesis", "autoeval_label", "raw_judge_response",
        "finish_reason", "judge_model", "token_usage",
        "cost_usd", "latency_s", "parse_warning",
    ]
    for i, r in enumerate(eval_results):
        for field in required_fields:
            if field not in r:
                failures.append(
                    f"eval result {i} ({r.get('question_id', '?')}) "
                    f"missing field '{field}'"
                )
    if not any("missing field" in f for f in failures):
        print("    OK: All eval results have required fields")

    # --- eval-summary ---
    if not eval_summary_path.exists():
        failures.append(f"Missing {eval_summary_path}")
        return failures, eval_results, None
    with open(eval_summary_path) as f:
        eval_summary: dict = json.load(f)
    print("    OK: eval-summary.json parseable")

    return failures, eval_results, eval_summary


# ── 2. Recompute summary consistency ──────────────────────────────────

def validate_summary_consistency(
    eval_results: list,
    eval_summary: dict,
) -> List[str]:
    """Recompute aggregate stats from eval-results and check against summary."""
    failures: List[str] = []

    correct = sum(1 for r in eval_results if r.get("autoeval_label") is True)
    incorrect = sum(1 for r in eval_results if r.get("autoeval_label") is False)
    errors = sum(1 for r in eval_results if r.get("autoeval_label") is None)
    total = correct + incorrect + errors
    accuracy = correct / total if total > 0 else 0.0

    print(f"  Computed: total={total}, correct={correct}, "
          f"incorrect={incorrect}, errors={errors}, accuracy={accuracy:.4f}")

    def _cmp(key: str, computed: Any) -> None:
        summary_val = eval_summary.get(key)
        if summary_val != computed:
            failures.append(f"{key}: summary={summary_val}, computed={computed}")
        else:
            print(f"    OK: {key}={computed}")

    _cmp("total_items", total)
    _cmp("correct", correct)
    _cmp("incorrect", incorrect)
    _cmp("errors", errors)
    _cmp("accuracy", round(accuracy, 4))

    # by_question_type
    by_type_summary: dict = eval_summary.get("by_question_type", {})
    by_type_computed: Dict[str, dict] = {}
    for r in eval_results:
        qt = r.get("question_type", "unknown")
        if qt not in by_type_computed:
            by_type_computed[qt] = {
                "correct": 0, "incorrect": 0, "errors": 0, "count": 0
            }
        by_type_computed[qt]["count"] += 1
        label = r.get("autoeval_label")
        if label is True:
            by_type_computed[qt]["correct"] += 1
        elif label is False:
            by_type_computed[qt]["incorrect"] += 1
        else:
            by_type_computed[qt]["errors"] += 1

    for qt, stats in by_type_computed.items():
        total_qt = stats["count"]
        acc_qt = stats["correct"] / total_qt if total_qt > 0 else 0.0
        if qt not in by_type_summary:
            failures.append(
                f"Question type '{qt}' missing from summary by_question_type"
            )
            continue
        s = by_type_summary[qt]
        if s.get("count") != total_qt:
            failures.append(
                f"{qt}: count summary={s.get('count')}, computed={total_qt}"
            )
        else:
            print(f"    OK: {qt} count={total_qt}")
        if abs(s.get("accuracy", 0) - acc_qt) > 0.0001:
            failures.append(
                f"{qt}: accuracy summary={s.get('accuracy'):.4f}, "
                f"computed={acc_qt:.4f}"
            )
        else:
            print(f"    OK: {qt} accuracy={acc_qt:.4f} ({acc_qt*100:.1f}%)")

    for qt in by_type_summary:
        if qt not in by_type_computed:
            failures.append(
                f"Question type '{qt}' in summary but not in eval-results"
            )

    return failures


# ── 3. SHA256 verification against PROVENANCE (fail-closed) ───────────

def parse_provenance_hashes(provenance_path: Path) -> Dict[str, str]:
    """Extract {relative_path: sha256hex} from a PROVENANCE.md file.

    Returns an empty dict when the table is present but no hashes can be
    parsed — callers should treat this as a validation failure.
    """
    hashes: Dict[str, str] = {}
    if not provenance_path.exists():
        return hashes  # empty → caller treats as failure
    with open(provenance_path) as f:
        text = f.read()
    for line in text.splitlines():
        match = re.search(
            r'`?(results/[^`\s]+)`?\s*\|\s*`?([a-f0-9]{64})`?',
            line,
        )
        if match:
            hashes[match.group(1)] = match.group(2)
    return hashes


def validate_provenance(
    provenance_path: Path,
    results_dir: Path,
    provenance_hashes: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Validate SHA-256 integrity of artifact files against PROVENANCE.md.

    Fail-closed: missing PROVENANCE or zero parsed hashes are hard failures.
    If *provenance_hashes* is supplied (e.g. from a pre-parse call), it is used
    directly instead of re-reading the file.
    """
    failures: List[str] = []

    if provenance_hashes is None:
        provenance_hashes = parse_provenance_hashes(provenance_path)

    if not provenance_path.exists():
        failures.append(
            f"PROVENANCE.md not found at {provenance_path} — "
            f"cannot verify artifact integrity"
        )
        return failures

    if not provenance_hashes:
        failures.append(
            "No SHA256 hashes found in PROVENANCE.md — "
            "cannot verify artifact integrity"
        )
        return failures

    for file_rel, expected_hash in provenance_hashes.items():
        file_path = results_dir.parent / file_rel
        if not file_path.exists():
            failures.append(
                f"PROVENANCE references '{file_rel}' but file not found "
                f"at {file_path}"
            )
            continue
        actual_hash = file_sha256(file_path)
        if actual_hash != expected_hash:
            failures.append(
                f"{file_rel}: SHA256 MISMATCH\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )
        else:
            print(f"    OK: {file_rel}: {expected_hash[:16]}...")

    return failures


# ── Orchestration ─────────────────────────────────────────────────────

def run_all_checks(base: Path) -> List[str]:
    """Run all validation checks, return aggregated failure list.

    Each check prints progress to stdout; failures are collected into a flat
    list returned to the caller (no global mutable state).
    """
    paths = resolve_paths(base)
    all_failures: List[str] = []

    # 1. Parseability & counts
    print("\n=== 1. Parseability and Counts ===")
    f1, eval_results, eval_summary = validate_parseability_and_counts(
        paths["hypotheses"],
        paths["eval_results"],
        paths["eval_summary"],
    )
    all_failures.extend(f1)
    if eval_results is None or eval_summary is None:
        return all_failures  # can't continue

    # 2. Summary consistency
    print("\n=== 2. Summary Consistency ===")
    all_failures.extend(validate_summary_consistency(eval_results, eval_summary))

    # 3. Provenance (fail-closed)
    print("\n=== 3. SHA256 Integrity (PROVENANCE.md) ===")
    all_failures.extend(
        validate_provenance(paths["provenance"], paths["results_dir"])
    )

    return all_failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Native LongMemEval Artifact Integrity Validator"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE,
        help="Root directory containing PROVENANCE.md and results/",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Native LongMemEval Artifact Integrity Validator")
    print("=" * 60)

    failures = run_all_checks(args.base_dir)

    print("\n" + "=" * 60)
    if failures:
        for msg in failures:
            print(f"FAIL: {msg}")
        print(f"\n{len(failures)} failure(s) — CHECKS FAILED")
        print("=" * 60)
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
