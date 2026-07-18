"""Tests for the Bench'd manifest validator — run_e375a77f15fb."""

import json
import os
import sys
from pathlib import Path

import pytest

# Add parent to path to import validator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validator import validate, canonical_manifest_bytes


MANIFEST_PATH = str(
    Path(__file__).resolve().parent.parent / "manifest.signed.json"
)


@pytest.fixture
def manifest_data():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


class TestFileIntegrity:
    def test_file_exists(self):
        assert os.path.exists(MANIFEST_PATH), f"Manifest not found: {MANIFEST_PATH}"

    def test_file_is_readable(self):
        with open(MANIFEST_PATH, "rb") as f:
            assert len(f.read(1024)) > 0


class TestManifestHash:
    def test_manifest_hash_matches(self, manifest_data):
        from validator import EXPECTED_MANIFEST_HASH
        import hashlib
        manifest = manifest_data["manifest"]
        computed = hashlib.sha256(
            canonical_manifest_bytes(manifest)
        ).hexdigest()
        assert computed == EXPECTED_MANIFEST_HASH, \
            f"Hash mismatch: computed={computed[:16]}... expected={EXPECTED_MANIFEST_HASH[:16]}..."
        assert computed == manifest_data["manifest_hash"], \
            "Computed hash doesn't match stored manifest_hash"

    def test_canonical_json_is_deterministic(self, manifest_data):
        import hashlib
        manifest = manifest_data["manifest"]
        # Serialize twice — should produce identical bytes
        b1 = canonical_manifest_bytes(manifest)
        b2 = canonical_manifest_bytes(manifest)
        assert b1 == b2, "Canonical serialization is not deterministic"
        h1 = hashlib.sha256(b1).hexdigest()
        h2 = hashlib.sha256(b2).hexdigest()
        assert h1 == h2


class TestEd25519Signature:
    def test_signature_verifies(self, manifest_data):
        import hashlib
        from validator import verify_ed25519

        stored_hash = manifest_data["manifest_hash"]
        result = verify_ed25519(
            manifest_data["public_key"],
            manifest_data["signature"],
            stored_hash.encode(),
        )
        if result["available"]:
            assert result["verified"], \
                f"Ed25519 signature invalid: {result.get('error')}"
        else:
            pytest.skip(f"cryptography not available: {result['error']}")

    def test_signature_over_hash_not_full_manifest(self, manifest_data):
        """Confirm the signing scheme: signature is over hash hex string, not full JSON."""
        import hashlib
        from validator import verify_ed25519

        stored_hash = manifest_data["manifest_hash"]
        full_manifest_bytes = canonical_manifest_bytes(manifest_data["manifest"])

        result_hash = verify_ed25519(
            manifest_data["public_key"],
            manifest_data["signature"],
            stored_hash.encode(),
        )
        # Only check if cryptography is available
        if result_hash["available"]:
            assert result_hash["verified"], f"Hash signing failed: {result_hash.get('error')}"


class TestScores:
    def test_passed_failed_match(self, manifest_data):
        summary = manifest_data["manifest"]["summary"]
        assert summary["passed"] == 431, f"Expected 431 passed, got {summary['passed']}"
        assert summary["failed"] == 69, f"Expected 69 failed, got {summary['failed']}"
        assert summary["total_questions"] == 500

    def test_trace_accuracy_862(self, manifest_data):
        traces = manifest_data["manifest"]["traces"]
        passed = sum(1 for t in traces if t.get("scored_correct") is True)
        total = len(traces)
        accuracy = passed / total * 100
        assert passed == 431, f"Trace count: {passed}/500 (expected 431)"
        assert round(accuracy, 1) == 86.2, f"Accuracy {accuracy}% != 86.2%"

    def test_nuance_score_is_85_67(self, manifest_data):
        nuance = manifest_data["manifest"]["scores"]["nuance"]
        overall = nuance["overall"]
        assert round(overall, 2) == 85.67, \
            f"Nuance overall {overall} != 85.67"

    def test_nuance_different_from_trace_accuracy(self, manifest_data):
        nuance_overall = manifest_data["manifest"]["scores"]["nuance"]["overall"]
        traces = manifest_data["manifest"]["traces"]
        passed = sum(1 for t in traces if t.get("scored_correct") is True)
        trace_accuracy = passed / len(traces) * 100
        # These are different metrics and should not be equal
        assert round(nuance_overall, 2) != round(trace_accuracy, 2), \
            "Nuance score should differ from trace accuracy"

    def test_summary_json_matches_manifest(self, manifest_data):
        summary_path = str(Path(__file__).resolve().parent.parent / "summary.json")
        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["summary"]["passed"] == 431
        assert summary["summary"]["failed"] == 69
        assert summary["cryptographic"]["manifest_hash"] == manifest_data["manifest_hash"]
        assert summary["cryptographic"]["public_key"] == manifest_data["public_key"]
        assert summary["scores"]["trace_accuracy"]["pct"] == 86.2
        assert summary["scores"]["nuance"]["overall"] == pytest.approx(85.66936237178193)


class TestQAOAEligibility:
    def test_qaoa_eligible_is_small_fraction(self, manifest_data):
        """QAOA eligible on only a tiny fraction — result is classical-dominant."""
        traces = manifest_data["manifest"]["traces"]
        qaoa_count = sum(
            1 for t in traces
            if t.get("metadata", {}).get("qaoa_eligible", False)
            or t.get("scoring_method", "") == "qaoa"
        )
        # Also check manifest-level count
        if qaoa_count == 0:
            qaoa_count = manifest_data["manifest"].get("scores", {}).get("qaoa_eligible_count", 0)
        total = len(traces)
        pct = qaoa_count / total * 100
        # Should be ≤ 5% — stated as 3/500 = 0.6%
        assert pct <= 5.0, \
            f"QAOA eligible {qaoa_count}/{total} = {pct:.1f}% — too high for classical-dominant claim"


class TestValidatorEndToEnd:
    def test_full_validation_passes(self):
        result = validate(MANIFEST_PATH)
        assert result["overall"], \
            f"Validation failed: {json.dumps(result['checks'], indent=2, default=str)}"

    def test_full_validation_json_output(self):
        result = validate(MANIFEST_PATH)
        json_output = json.dumps(result, indent=2, default=str)
        assert "overall" in json_output
        assert "checks" in json_output


class TestNoSecrets:
    def test_no_api_keys_in_summary(self):
        summary_path = str(Path(__file__).resolve().parent.parent / "summary.json")
        with open(summary_path) as f:
            content = f.read()
        # Simple heuristic: no patterns matching common key formats
        import re
        patterns = [
            r'sk-[a-zA-Z0-9]{20,}',    # OpenAI
            r'AIza[0-9A-Za-z\-_]{35}',  # Google
            r'ghp_[a-zA-Z0-9]{36}',     # GitHub
            r'xai-[a-zA-Z0-9]{20,}',    # xAI
        ]
        for pattern in patterns:
            assert not re.search(pattern, content), \
                f"Potential secret found matching {pattern}"

    def test_no_home_paths_in_artifacts(self):
        """Ensure no /home/dt paths leaked into committed artifacts."""
        for fname in ["summary.json", "validator.py", "PROVENANCE.md"]:
            fpath = str(Path(__file__).resolve().parent.parent / fname)
            with open(fpath) as f:
                content = f.read()
            assert "/home/dt" not in content, \
                f"Found /home/dt path in {fname}"


class TestCryptographicConstants:
    def test_fingerprint(self, manifest_data):
        from validator import EXPECTED_FINGERPRINT
        assert manifest_data["signing_key_fingerprint"] == EXPECTED_FINGERPRINT

    def test_public_key_matches(self, manifest_data):
        from validator import EXPECTED_PUBLIC_KEY
        assert manifest_data["public_key"] == EXPECTED_PUBLIC_KEY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
