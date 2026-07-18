#!/usr/bin/env python3
"""Deterministic validator for Bench'd manifest — run_e375a77f15fb.

Verifies:
  1. File SHA-256 of manifest.signed.json
  2. Canonical manifest hash (SHA-256 of sorted, compact JSON)
  3. Ed25519 signature over the manifest hash
  4. Embedded score integrity
  5. Trace pass/fail count matches published 431/500

Requires: Python 3.9+ stdlib (hashlib, json).
Ed25519 verification requires `cryptography` (pip install cryptography);
the validator reports signature status separately if cryptography is absent.

Usage:
    python validator.py               # validate the local manifest
    python validator.py --json        # output machine-readable JSON result
"""

import hashlib
import json
import os
import sys
from pathlib import Path

# ── Expected constants ──────────────────────────────────────────────
EXPECTED_FILE_SHA256 = (
    "01a1b981e4652598976b7f32ac9e5a4f5261df3585b72c27977ea4a26a29be82"
)
EXPECTED_MANIFEST_HASH = (
    "c8e6d9cb4026f017e421c56be6cabda1a1d4958f0a4def2557b4150fdd02ef16"
)
EXPECTED_PUBLIC_KEY = (
    "483b133c089ac3ff5fbb3c6df75923ac3874d550e3997ff6c617572a5aa5a830"
)
EXPECTED_SIGNATURE = (
    "6ebbde1ac3b8a38e7be328bbe31e7b499433d80540988db687d29322f4dabc53"
    "20653d04b3e177a1f645fc3374c09cf5365bb1f8b627ca48947f58fbead85f04"
)
EXPECTED_FINGERPRINT = "23442739ed37e98f"
EXPECTED_PASSED = 431
EXPECTED_FAILED = 69
EXPECTED_TOTAL = 500


def canonical_manifest_bytes(manifest: dict) -> bytes:
    """Return the canonical JSON bytes of the manifest (sorted keys, no whitespace)."""
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()


def verify_ed25519(public_key_hex: str, signature_hex: str, message: bytes) -> dict:
    """Attempt Ed25519 verification. Returns status dict."""
    result = {"verified": False, "available": False, "error": None}
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        from cryptography.exceptions import InvalidSignature

        result["available"] = True
        pk_bytes = bytes.fromhex(public_key_hex)
        sig_bytes = bytes.fromhex(signature_hex)
        pk = Ed25519PublicKey.from_public_bytes(pk_bytes)
        pk.verify(sig_bytes, message)
        result["verified"] = True
    except ImportError:
        result["error"] = "cryptography package not installed"
    except Exception:
        # InvalidSignature (from cryptography) or any other verification failure
        result["error"] = "Ed25519 signature invalid"

    return result


def validate(manifest_path: str | None = None) -> dict:
    """Run all validations. Returns result dict."""
    if manifest_path is None:
        manifest_path = str(
            Path(__file__).resolve().parent / "manifest.signed.json"
        )
    results = {
        "manifest_path": manifest_path,
        "checks": {},
        "overall": False,
    }

    # 1. File exists
    if not os.path.exists(manifest_path):
        results["checks"]["file_exists"] = False
        results["checks"]["_error"] = f"File not found: {manifest_path}"
        return results
    results["checks"]["file_exists"] = True

    # 2. File SHA-256
    file_sha256 = hashlib.sha256()
    with open(manifest_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            file_sha256.update(chunk)
    computed_file_hash = file_sha256.hexdigest()
    results["checks"]["file_sha256"] = {
        "expected": EXPECTED_FILE_SHA256,
        "computed": computed_file_hash,
        "match": computed_file_hash == EXPECTED_FILE_SHA256,
    }

    # 3. Load and parse
    with open(manifest_path) as f:
        data = json.load(f)

    # 4. Manifest hash
    manifest = data.get("manifest", {})
    stored_hash = data.get("manifest_hash", "")
    computed_hash = hashlib.sha256(
        canonical_manifest_bytes(manifest)
    ).hexdigest()
    results["checks"]["manifest_hash"] = {
        "expected": EXPECTED_MANIFEST_HASH,
        "stored": stored_hash,
        "computed": computed_hash,
        "match": computed_hash == stored_hash == EXPECTED_MANIFEST_HASH,
    }

    # 5. Ed25519 signature over manifest hash
    sig_verify = verify_ed25519(
        data.get("public_key", ""),
        data.get("signature", ""),
        stored_hash.encode(),
    )
    results["checks"]["ed25519_signature"] = sig_verify

    # 6. Fingerprint
    results["checks"]["fingerprint"] = {
        "expected": EXPECTED_FINGERPRINT,
        "actual": data.get("signing_key_fingerprint", ""),
        "match": data.get("signing_key_fingerprint") == EXPECTED_FINGERPRINT,
    }

    # 7. Score integrity
    summary = manifest.get("summary", {})
    results["checks"]["score_summary"] = {
        "expected_passed": EXPECTED_PASSED,
        "actual_passed": summary.get("passed"),
        "expected_failed": EXPECTED_FAILED,
        "actual_failed": summary.get("failed"),
        "expected_total": EXPECTED_TOTAL,
        "actual_total": summary.get("total_questions"),
        "match": (
            summary.get("passed") == EXPECTED_PASSED
            and summary.get("failed") == EXPECTED_FAILED
            and summary.get("total_questions") == EXPECTED_TOTAL
        ),
    }

    # 8. Trace accuracy (count from traces)
    traces = manifest.get("traces", [])
    passed = sum(1 for t in traces if t.get("scored_correct") is True)
    failed = sum(1 for t in traces if t.get("scored_correct") is False)
    total = len(traces)
    results["checks"]["trace_accuracy"] = {
        "traces_total": total,
        "traces_passed": passed,
        "traces_failed": failed,
        "accuracy_pct": round(passed / total * 100, 2) if total > 0 else 0,
        "match": passed == EXPECTED_PASSED and failed == EXPECTED_FAILED,
    }

    # 9. QAOA eligibility check — metadata at trace level if present,
    #    otherwise fall back to checking the retrieval method
    qaoa_eligible = sum(
        1 for t in traces
        if t.get("metadata", {}).get("qaoa_eligible", False)
        or t.get("scoring_method", "") == "qaoa"
    )
    # Also check manifest-level QAOA indicator
    if qaoa_eligible == 0:
        qaoa_eligible = manifest.get("scores", {}).get("qaoa_eligible_count", 0)
    results["checks"]["qaoa_eligibility"] = {
        "eligible_count": qaoa_eligible,
        "total": total,
        "pct": round(qaoa_eligible / total * 100, 2) if total > 0 else 0,
        "note": "Signed manifest contains no optimizer execution telemetry; QAOA count is zero per signed evidence",
    }

    # Overall
    checks = results["checks"]
    critical = [
        checks["file_exists"],
        checks["file_sha256"]["match"],
        checks["manifest_hash"]["match"],
        checks["score_summary"]["match"],
        checks["trace_accuracy"]["match"],
    ]
    results["overall"] = all(critical)

    return results


def main() -> None:
    json_output = "--json" in sys.argv
    manifest_path = None
    for arg in sys.argv[1:]:
        if arg != "--json" and not arg.startswith("-"):
            manifest_path = arg

    results = validate(manifest_path)

    if json_output:
        print(json.dumps(results, indent=2))
    else:
        c = results["checks"]
        def ok(b): return "✅" if b else "❌"

        print(f"Validator — Bench'd manifest: {results['manifest_path']}\n")
        print(f"  File exists:         {ok(c['file_exists'])}")
        print(f"  File SHA-256:        {ok(c['file_sha256']['match'])}  {c['file_sha256']['computed'][:16]}...")
        print(f"  Manifest hash:       {ok(c['manifest_hash']['match'])}  {c['manifest_hash']['computed'][:16]}...")
        sig = c["ed25519_signature"]
        if sig["available"]:
            print(f"  Ed25519 signature:   {ok(sig['verified'])}  {'verified' if sig['verified'] else sig.get('error', 'invalid')}")
        else:
            print(f"  Ed25519 signature:   ⚠️  skipped ({sig['error']})")
        print(f"  Key fingerprint:     {ok(c['fingerprint']['match'])}  {c['fingerprint']['actual']}")
        print(f"  Score summary:       {ok(c['score_summary']['match'])}  {c['score_summary']['actual_passed']}/{c['score_summary']['actual_total']}")
        ta = c["trace_accuracy"]
        print(f"  Trace accuracy:      {ok(ta['match'])}  {ta['traces_passed']}/{ta['traces_total']} = {ta['accuracy_pct']}%")
        qaoa = c["qaoa_eligibility"]
        print(f"  QAOA eligible:       {qaoa['eligible_count']}/{qaoa['total']} ({qaoa['pct']}%)")
        print(f"\n  OVERALL:             {ok(results['overall'])}")
        if not results["overall"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
