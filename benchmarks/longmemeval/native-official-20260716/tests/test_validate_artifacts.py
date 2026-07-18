#!/usr/bin/env python3
"""Dedicated unit tests for the native artifact validator.

All tests use temporary directories (tmp_path) — no global state leakage,
no dependency on the real artifact tree.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure the scripts/ directory is importable
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from validate_artifacts import (
    file_sha256,
    parse_provenance_hashes,
    resolve_paths,
    run_all_checks,
    validate_parseability_and_counts,
    validate_provenance,
    validate_summary_consistency,
)

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as f:
        f.write(content)


def _json_dump(path: Path, obj: Any) -> None:
    _write(path, json.dumps(obj))


def _make_minimal_eval_results(n: int = 3) -> list:
    """Return *n* well-formed eval-result dicts."""
    results: list = []
    for i in range(n):
        results.append({
            "question_id": f"q{i:04d}",
            "question_type": "single-session-user",
            "question": f"Q{i}?",
            "answer": f"A{i}",
            "hypothesis": f"H{i}",
            "autoeval_label": True if i % 2 == 0 else False,
            "raw_judge_response": "Yes." if i % 2 == 0 else "No.",
            "finish_reason": "stop",
            "judge_model": "openai/gpt-4o",
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            "cost_usd": 0.001,
            "latency_s": 0.5,
            "parse_warning": None,
        })
    return results


def _make_minimal_summary(
    total: int = 3,
    correct: int = 2,
    incorrect: int = 1,
    errors: int = 0,
    by_type: Dict[str, dict] | None = None,
) -> dict:
    acc = correct / total if total > 0 else 0.0
    if by_type is None:
        by_type = {
            "single-session-user": {"accuracy": acc, "count": total},
        }
    return {
        "total_items": total,
        "correct": correct,
        "incorrect": incorrect,
        "errors": errors,
        "accuracy": acc,
        "by_question_type": by_type,
    }


def _make_minimal_hypotheses(n: int = 3) -> list:
    return [
        {"question_id": f"q{i:04d}", "hypothesis": f"hypothesis-{i}"}
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════
# 1. resolve_paths
# ═══════════════════════════════════════════════════════════════════════

class TestResolvePaths:
    def test_produces_expected_keys(self, tmp_path: Path) -> None:
        paths = resolve_paths(tmp_path)
        for key in ("results_dir", "provenance", "eval_results",
                     "hypotheses", "eval_summary"):
            assert key in paths

    def test_paths_rooted_at_base(self, tmp_path: Path) -> None:
        paths = resolve_paths(tmp_path)
        assert paths["results_dir"] == tmp_path / "results"
        assert paths["provenance"] == tmp_path / "PROVENANCE.md"


# ═══════════════════════════════════════════════════════════════════════
# 2. file_sha256
# ═══════════════════════════════════════════════════════════════════════

class TestFileSha256:
    def test_known_content(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        _write(f, "hello world\n")
        expected = hashlib.sha256(b"hello world\n").hexdigest()
        assert file_sha256(f) == expected

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty"
        _write(f, "")
        assert file_sha256(f) == hashlib.sha256(b"").hexdigest()


# ═══════════════════════════════════════════════════════════════════════
# 3. parse_provenance_hashes
# ═══════════════════════════════════════════════════════════════════════

class TestParseProvenanceHashes:
    HASH = "a" * 64

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert parse_provenance_hashes(tmp_path / "nope.md") == {}

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "PROVENANCE.md"
        _write(p, "")
        assert parse_provenance_hashes(p) == {}

    def test_no_sha_rows_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "PROVENANCE.md"
        _write(p, "# Just text\n\nno hashes here\n")
        assert parse_provenance_hashes(p) == {}

    def test_parses_valid_table(self, tmp_path: Path) -> None:
        p = tmp_path / "PROVENANCE.md"
        _write(p, (
            "| File | SHA-256 | Description |\n"
            "|------|---------|-------------|\n"
            f"| `results/x.json` | `{self.HASH}` | desc |\n"
        ))
        h = parse_provenance_hashes(p)
        assert h == {"results/x.json": self.HASH}

    def test_parses_multiple_entries(self, tmp_path: Path) -> None:
        p = tmp_path / "PROVENANCE.md"
        h1 = "b" * 64
        h2 = "c" * 64
        _write(p, (
            "| File | SHA-256 | Description |\n"
            "|------|---------|-------------|\n"
            f"| `results/a.json` | `{h1}` | |\n"
            f"| `results/b.jsonl` | `{h2}` | |\n"
        ))
        h = parse_provenance_hashes(p)
        assert len(h) == 2
        assert h["results/a.json"] == h1
        assert h["results/b.jsonl"] == h2

    def test_ignores_incomplete_rows(self, tmp_path: Path) -> None:
        p = tmp_path / "PROVENANCE.md"
        _write(p, (
            "| `results/x.json` | `aaa` |\n"   # short hash
        ))
        assert parse_provenance_hashes(p) == {}

    def test_malformed_provenance_is_non_fatal(self, tmp_path: Path) -> None:
        """Malformed markdown should just yield empty dict, not crash."""
        p = tmp_path / "PROVENANCE.md"
        _write(p, "garbage without any table\n")
        assert parse_provenance_hashes(p) == {}


# ═══════════════════════════════════════════════════════════════════════
# 4. validate_provenance  (fail-closed)
# ═══════════════════════════════════════════════════════════════════════

class TestValidateProvenance:
    HASH = "a" * 64

    def test_missing_file_is_failure(self, tmp_path: Path) -> None:
        failures = validate_provenance(
            tmp_path / "nonexistent.md",
            tmp_path / "results",
        )
        assert len(failures) >= 1
        assert any("not found" in f.lower() for f in failures)

    def test_no_parsed_hashes_is_failure(self, tmp_path: Path) -> None:
        prov = tmp_path / "PROVENANCE.md"
        _write(prov, "# No hashes\n")
        failures = validate_provenance(prov, tmp_path / "results")
        assert len(failures) >= 1
        assert any("no sha256" in f.lower() for f in failures)

    def test_empty_hashes_dict_is_failure(self, tmp_path: Path) -> None:
        prov = tmp_path / "PROVENANCE.md"
        _write(prov, "placeholder")
        results = tmp_path / "results"
        results.mkdir()
        failures = validate_provenance(prov, results, provenance_hashes={})
        assert len(failures) >= 1

    def test_all_matching_passes(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        results.mkdir()
        f = results / "x.json"
        _write(f, "data")
        prov = tmp_path / "PROVENANCE.md"
        h = file_sha256(f)
        _write(prov, (
            "| File | SHA-256 | Description |\n"
            "|------|---------|-------------|\n"
            f"| `results/x.json` | `{h}` | |\n"
        ))
        failures = validate_provenance(prov, results)
        assert failures == []

    def test_hash_mismatch_is_failure(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        results.mkdir()
        f = results / "x.json"
        _write(f, "original")
        prov = tmp_path / "PROVENANCE.md"
        wrong_hash = "b" * 64
        _write(prov, (
            "| File | SHA-256 | Description |\n"
            "|------|---------|-------------|\n"
            f"| `results/x.json` | `{wrong_hash}` | |\n"
        ))
        failures = validate_provenance(prov, results)
        assert len(failures) >= 1
        assert any("mismatch" in f.lower() for f in failures)

    def test_referenced_file_missing_is_failure(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        results.mkdir()
        prov = tmp_path / "PROVENANCE.md"
        _write(prov, (
            "| File | SHA-256 | Description |\n"
            "|------|---------|-------------|\n"
            f"| `results/ghost.json` | `{self.HASH}` | |\n"
        ))
        failures = validate_provenance(prov, results)
        assert len(failures) >= 1
        assert any("not found" in f.lower() for f in failures)


# ═══════════════════════════════════════════════════════════════════════
# 5. validate_parseability_and_counts
# ═══════════════════════════════════════════════════════════════════════

class TestParseabilityAndCounts:
    def test_all_good(self, tmp_path: Path) -> None:
        hyps = tmp_path / "hypotheses.jsonl"
        _write(hyps, "\n".join(
            json.dumps(h) for h in _make_minimal_hypotheses(500)
        ))
        er = tmp_path / "eval_results.json"
        _json_dump(er, _make_minimal_eval_results(500))
        es = tmp_path / "summary.json"
        _json_dump(es, _make_minimal_summary(500, 250, 250))
        failures, results, summary = validate_parseability_and_counts(hyps, er, es)
        assert failures == []
        assert len(results) == 500
        assert summary is not None

    def test_missing_hypotheses(self, tmp_path: Path) -> None:
        hyps = tmp_path / "nope.jsonl"
        failures, results, summary = validate_parseability_and_counts(
            hyps, tmp_path / "er.json", tmp_path / "es.json"
        )
        assert any("missing" in f.lower() for f in failures)
        assert results is None

    def test_missing_eval_results(self, tmp_path: Path) -> None:
        hyps = tmp_path / "h.jsonl"
        _write(hyps, "\n".join(json.dumps(h) for h in _make_minimal_hypotheses(500)))
        failures, results, summary = validate_parseability_and_counts(
            hyps, tmp_path / "nope.json", tmp_path / "es.json"
        )
        assert any("missing" in f.lower() for f in failures)
        assert results is None

    def test_missing_summary(self, tmp_path: Path) -> None:
        hyps = tmp_path / "h.jsonl"
        _write(hyps, "\n".join(json.dumps(h) for h in _make_minimal_hypotheses(500)))
        er = tmp_path / "er.json"
        _json_dump(er, _make_minimal_eval_results(500))
        failures, results, summary = validate_parseability_and_counts(
            hyps, er, tmp_path / "nope.json"
        )
        assert any("missing" in f.lower() for f in failures)
        assert summary is None

    def test_wrong_hypothesis_count(self, tmp_path: Path) -> None:
        hyps = tmp_path / "h.jsonl"
        _write(hyps, "\n".join(json.dumps(h) for h in _make_minimal_hypotheses(3)))
        er = tmp_path / "er.json"
        _json_dump(er, _make_minimal_eval_results(500))
        es = tmp_path / "es.json"
        _json_dump(es, _make_minimal_summary())
        failures, _, _ = validate_parseability_and_counts(hyps, er, es)
        assert any("expected 500" in f.lower() for f in failures)

    def test_wrong_eval_results_count(self, tmp_path: Path) -> None:
        hyps = tmp_path / "h.jsonl"
        _write(hyps, "\n".join(json.dumps(h) for h in _make_minimal_hypotheses(500)))
        er = tmp_path / "er.json"
        _json_dump(er, _make_minimal_eval_results(3))
        es = tmp_path / "es.json"
        _json_dump(es, _make_minimal_summary())
        failures, _, _ = validate_parseability_and_counts(hyps, er, es)
        assert any("expected 500" in f.lower() for f in failures)

    def test_malformed_jsonl(self, tmp_path: Path) -> None:
        hyps = tmp_path / "h.jsonl"
        _write(hyps, '{"question_id":"q0","hypothesis":"h0"}\nnot json\n')
        er = tmp_path / "er.json"
        _json_dump(er, _make_minimal_eval_results(500))
        es = tmp_path / "es.json"
        _json_dump(es, _make_minimal_summary())
        failures, _, _ = validate_parseability_and_counts(hyps, er, es)
        assert any("invalid json" in f.lower() for f in failures)

    def test_malformed_json_eval_results(self, tmp_path: Path) -> None:
        hyps = tmp_path / "h.jsonl"
        _write(hyps, "\n".join(json.dumps(h) for h in _make_minimal_hypotheses(500)))
        er = tmp_path / "er.json"
        _write(er, "this is not json")
        es = tmp_path / "es.json"
        _json_dump(es, _make_minimal_summary())
        failures, results, summary = validate_parseability_and_counts(hyps, er, es)
        assert results is None
        assert summary is None
        assert any("invalid json" in f.lower() for f in failures)

    def test_corrupted_eval_results_json(self, tmp_path: Path) -> None:
        """Corrupted eval-results JSON produces graceful failure (no crash)."""
        hyps = tmp_path / "h.jsonl"
        _write(hyps, "\n".join(json.dumps(h) for h in _make_minimal_hypotheses(500)))
        er = tmp_path / "er.json"
        _write(er, "garbage{{{{")
        es = tmp_path / "es.json"
        _json_dump(es, _make_minimal_summary())
        failures, results, summary = validate_parseability_and_counts(hyps, er, es)
        assert results is None
        assert any("invalid json" in f.lower() for f in failures)

    def test_missing_required_field_in_hypothesis(self, tmp_path: Path) -> None:
        hyps = tmp_path / "h.jsonl"
        _write(hyps, '{"question_id":"q0"}\n')  # missing hypothesis
        er = tmp_path / "er.json"
        _json_dump(er, _make_minimal_eval_results(500))
        es = tmp_path / "es.json"
        _json_dump(es, _make_minimal_summary())
        failures, _, _ = validate_parseability_and_counts(hyps, er, es)
        assert any("missing field" in f.lower() for f in failures)

    def test_missing_required_field_in_eval_result(self, tmp_path: Path) -> None:
        hyps = tmp_path / "h.jsonl"
        _write(hyps, "\n".join(json.dumps(h) for h in _make_minimal_hypotheses(500)))
        er = tmp_path / "er.json"
        results = _make_minimal_eval_results(500)
        del results[0]["question_id"]
        _json_dump(er, results)
        es = tmp_path / "es.json"
        _json_dump(es, _make_minimal_summary())
        failures, _, _ = validate_parseability_and_counts(hyps, er, es)
        assert any("missing field" in f.lower() for f in failures)

    def test_eval_results_not_array(self, tmp_path: Path) -> None:
        hyps = tmp_path / "h.jsonl"
        _write(hyps, "\n".join(json.dumps(h) for h in _make_minimal_hypotheses(500)))
        er = tmp_path / "er.json"
        _json_dump(er, {"not": "an array"})
        es = tmp_path / "es.json"
        _json_dump(es, _make_minimal_summary())
        failures, results, _ = validate_parseability_and_counts(hyps, er, es)
        assert any("not a json array" in f.lower() for f in failures)
        assert isinstance(results, dict)  # non-list results returned as-is


# ═══════════════════════════════════════════════════════════════════════
# 6. validate_summary_consistency
# ═══════════════════════════════════════════════════════════════════════

class TestSummaryConsistency:
    def test_perfect_match(self) -> None:
        results = _make_minimal_eval_results(5)
        # 3 True, 2 False
        for i in range(3):
            results[i]["autoeval_label"] = True
        for i in range(3, 5):
            results[i]["autoeval_label"] = False
        summary = _make_minimal_summary(5, 3, 2, 0,
                                        by_type={"single-session-user": {"accuracy": 0.6, "count": 5}})
        failures = validate_summary_consistency(results, summary)
        assert failures == []

    def test_mismatched_total(self) -> None:
        results = _make_minimal_eval_results(3)
        summary = _make_minimal_summary(9, 5, 4, 0)
        failures = validate_summary_consistency(results, summary)
        assert any("total_items" in f for f in failures)

    def test_mismatched_correct(self) -> None:
        results = _make_minimal_eval_results(3)
        summary = _make_minimal_summary(3, 99, 0, 0)
        failures = validate_summary_consistency(results, summary)
        assert any("correct" in f for f in failures)

    def test_mismatched_accuracy(self) -> None:
        results = _make_minimal_eval_results(3)
        # 2 correct out of 3 = 0.6666...
        results[0]["autoeval_label"] = True
        results[1]["autoeval_label"] = True
        results[2]["autoeval_label"] = False
        summary = _make_minimal_summary(3, 2, 1, 0,
                                        by_type={"single-session-user": {"accuracy": 2.0, "count": 3}})
        failures = validate_summary_consistency(results, summary)
        assert any("accuracy" in f for f in failures)

    def test_missing_question_type_in_summary(self) -> None:
        results = _make_minimal_eval_results(3)
        results[0]["autoeval_label"] = True
        results[1]["autoeval_label"] = True
        results[2]["autoeval_label"] = False
        summary = _make_minimal_summary(3, 2, 1, 0, by_type={})
        failures = validate_summary_consistency(results, summary)
        assert any("missing from summary" in f for f in failures)

    def test_extra_question_type_in_summary(self) -> None:
        results = _make_minimal_eval_results(3)
        results[0]["autoeval_label"] = True
        results[1]["autoeval_label"] = True
        results[2]["autoeval_label"] = False
        summary = _make_minimal_summary(3, 2, 1, 0,
                                        by_type={
                                            "single-session-user": {"accuracy": 0.6667, "count": 3},
                                            "ghost-type": {"accuracy": 0.0, "count": 0},
                                        })
        failures = validate_summary_consistency(results, summary)
        assert any("in summary but not in eval-results" in f for f in failures)

    def test_all_errors_corner_case(self) -> None:
        """All items have autoeval_label=None (errors)."""
        results = _make_minimal_eval_results(3)
        for r in results:
            r["autoeval_label"] = None
        summary = _make_minimal_summary(3, 0, 0, 3,
                                        by_type={"single-session-user": {"accuracy": 0.0, "count": 3}})
        failures = validate_summary_consistency(results, summary)
        assert failures == []


# ═══════════════════════════════════════════════════════════════════════
# 7. run_all_checks  (integration-style, still tmp_path)
# ═══════════════════════════════════════════════════════════════════════

class TestRunAllChecks:
    """End-to-end checks using a fully populated tmp_path directory tree."""

    def _setup_full_artifacts(self, base: Path, n: int = 500) -> None:
        """Create all expected artifact files under *base*."""
        results = base / "results"
        results.mkdir(parents=True)

        hyps = [
            {"question_id": f"q{i:04d}", "hypothesis": f"hypothesis-{i}"}
            for i in range(n)
        ]
        _write(results / "hypotheses_merged_500.jsonl",
               "\n".join(json.dumps(h) for h in hyps))

        eval_results = _make_minimal_eval_results(n)
        correct = sum(1 for r in eval_results if r["autoeval_label"] is True)
        incorrect = n - correct
        _json_dump(results / "eval-results-500.json", eval_results)

        summary = {
            "total_items": n,
            "correct": correct,
            "incorrect": incorrect,
            "errors": 0,
            "accuracy": correct / n,
            "by_question_type": {
                "single-session-user": {
                    "accuracy": correct / n, "count": n
                },
            },
        }
        _json_dump(results / "eval-summary.json", summary)

        # PROVENANCE
        h_hyps = file_sha256(results / "hypotheses_merged_500.jsonl")
        h_er = file_sha256(results / "eval-results-500.json")
        h_es = file_sha256(results / "eval-summary.json")
        _write(base / "PROVENANCE.md", (
            "| File | SHA-256 | Description |\n"
            "|------|---------|-------------|\n"
            f"| `results/hypotheses_merged_500.jsonl` | `{h_hyps}` | |\n"
            f"| `results/eval-results-500.json` | `{h_er}` | |\n"
            f"| `results/eval-summary.json` | `{h_es}` | |\n"
        ))

    def test_all_pass(self, tmp_path: Path) -> None:
        self._setup_full_artifacts(tmp_path)
        failures = run_all_checks(tmp_path)
        assert failures == [], f"Unexpected failures: {failures}"

    def test_missing_provenance_fails(self, tmp_path: Path) -> None:
        self._setup_full_artifacts(tmp_path)
        (tmp_path / "PROVENANCE.md").unlink()
        failures = run_all_checks(tmp_path)
        assert any("not found" in f.lower() for f in failures)

    def test_hash_mismatch_fails(self, tmp_path: Path) -> None:
        self._setup_full_artifacts(tmp_path)
        # Corrupt one artifact (change content, not JSON validity)
        results = tmp_path / "results"
        _write(results / "eval-results-500.json",
               json.dumps(_make_minimal_eval_results(3)))  # valid JSON, wrong count
        failures = run_all_checks(tmp_path)
        assert any("mismatch" in f.lower() for f in failures)

    def test_wrong_count_fails(self, tmp_path: Path) -> None:
        self._setup_full_artifacts(tmp_path)
        results = tmp_path / "results"
        _write(results / "hypotheses_merged_500.jsonl",
               "\n".join(json.dumps({"question_id": f"q{i:04d}", "hypothesis": f"h{i}"})
                         for i in range(3)))
        failures = run_all_checks(tmp_path)
        assert any("expected 500" in f.lower() for f in failures)

    def test_no_parsed_hashes_fails(self, tmp_path: Path) -> None:
        self._setup_full_artifacts(tmp_path)
        # Replace PROVENANCE with useless content
        _write(tmp_path / "PROVENANCE.md", "No table here.\n")
        failures = run_all_checks(tmp_path)
        assert any("no sha256" in f.lower() for f in failures)

    def test_malformed_hypotheses_fails(self, tmp_path: Path) -> None:
        self._setup_full_artifacts(tmp_path)
        results = tmp_path / "results"
        _write(results / "hypotheses_merged_500.jsonl", "garbage\nnot json\n")
        failures = run_all_checks(tmp_path)
        assert any("invalid json" in f.lower() for f in failures)

    def test_missing_field_fails(self, tmp_path: Path) -> None:
        self._setup_full_artifacts(tmp_path)
        results = tmp_path / "results"
        eval_list = _make_minimal_eval_results(500)
        del eval_list[0]["question_id"]
        _json_dump(results / "eval-results-500.json", eval_list)
        failures = run_all_checks(tmp_path)
        assert any("missing field" in f.lower() for f in failures)

    def test_summary_mismatch_fails(self, tmp_path: Path) -> None:
        self._setup_full_artifacts(tmp_path)
        results = tmp_path / "results"
        summary = json.loads((results / "eval-summary.json").read_text())
        summary["total_items"] = 999
        _json_dump(results / "eval-summary.json", summary)
        failures = run_all_checks(tmp_path)
        assert any("total_items" in f for f in failures)


# ═══════════════════════════════════════════════════════════════════════
# 8. Edge cases: duplicate / mismatched question IDs
# ═══════════════════════════════════════════════════════════════════════

class TestQuestionIdEdgeCases:
    """Ensure validator doesn't crash on duplicate or mismatched IDs."""

    def test_duplicate_question_ids_dont_crash(self, tmp_path: Path) -> None:
        """Duplicate IDs may be questionable but shouldn't crash the validator."""
        hyps = tmp_path / "h.jsonl"
        _write(hyps, "\n".join(
            json.dumps({"question_id": "dup", "hypothesis": f"h{i}"})
            for i in range(500)
        ))
        er = tmp_path / "er.json"
        _json_dump(er, _make_minimal_eval_results(500))
        es = tmp_path / "es.json"
        _json_dump(es, _make_minimal_summary(500, 250, 250))
        failures, _, _ = validate_parseability_and_counts(hyps, er, es)
        # Count check should still pass (500 entries), no crash
        assert not any("expected 500" in f.lower() for f in failures)

    def test_mismatched_ids_across_files_dont_crash(self, tmp_path: Path) -> None:
        """Question IDs in hypotheses vs eval-results may differ — not fatal."""
        hyps = tmp_path / "h.jsonl"
        _write(hyps, "\n".join(
            json.dumps({"question_id": f"hyp_{i:04d}", "hypothesis": f"h{i}"})
            for i in range(500)
        ))
        er = tmp_path / "er.json"
        _json_dump(er, _make_minimal_eval_results(500))
        es = tmp_path / "es.json"
        _json_dump(es, _make_minimal_summary(500, 250, 250))
        failures, _, _ = validate_parseability_and_counts(hyps, er, es)
        # No crash; may or may not produce ID-specific failures
        assert isinstance(failures, list)
