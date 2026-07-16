#!/usr/bin/env python3
"""
LongMemEval QA Evaluator — OpenRouter GPT-4o wrapper.

Imports the ***exact*** get_anscheck_prompt from the official LongMemEval
evaluator (`LongMemEval/src/evaluation/evaluate_qa.py`, commit 9e0b455),
then calls OpenRouter GPT-4o (openai/gpt-4o) with an OpenAI-compatible
client over the runtime credential file `./benchd-openrouter.env`.

Key differences from official evaluate_qa.py:
- Judge via OpenRouter instead of direct OpenAI
- Strict substring-bug fix: only 'yes' / 'no' after normalized exact
  match counts; 'yesterday' does NOT match 'yes'
- Fail-closed: empty, ambiguous, or unparseable judge output → label=False
- Rich per-item metadata: raw response, token usage, cost, hashes

CREDENTIAL SAFETY: The API key is loaded from the env file at runtime and
NEVER echoed, logged, or stored — not even a prefix. No portion of the key
ever appears on stdout, stderr, or any output artifact.

Usage:
    python3 evaluate_qa_openrouter.py \
        --hyp-file runs/.../qmg_retrieval_10_hypotheses_*.jsonl \
        --ref-file LongMemEval/data/longmemeval_s_cleaned.json \
        --out-dir eval-wrapper/results
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import backoff
import openai
from openai import OpenAI

# ── Import the EXACT get_anscheck_prompt from official evaluator ─────
REPO_ROOT = Path(__file__).resolve().parent.parent
LONGMEMEVAL_DIR = REPO_ROOT / "LongMemEval"
OFFICIAL_EVAL_DIR = LONGMEMEVAL_DIR / "src" / "evaluation"

if str(OFFICIAL_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_EVAL_DIR))

# Import the exact function — no copy-paste, no drift
from evaluate_qa import get_anscheck_prompt  # type: ignore[import]

# ── Constants ────────────────────────────────────────────────────────
JUDGE_MODEL = "openai/gpt-4o"
OFFICIAL_COMMIT = "9e0b455"  # Update me if the official repo moves
CRED_FILE = Path("./benchd-openrouter.env")  # --cred-file

# OpenRouter pricing (per 1M tokens, as of July 2025)
# GPT-4o via OpenRouter: $2.50/M input, $10.00/M output
PRICE_PER_1M_INPUT = 2.50
PRICE_PER_1M_OUTPUT = 10.00


# ── Credential loading ───────────────────────────────────────────────
def load_creds(cred_file: Path) -> tuple[str, str]:
    """Load OpenRouter base_url and API key from env file.

    Returns (base_url, api_key). NEVER prints the key.
    """
    env_vars = {}
    if not cred_file.exists():
        raise FileNotFoundError(f"Credential file not found: {cred_file}")
    with open(cred_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                env_vars[key.strip()] = val.strip().strip('"').strip("'")

    base_url = env_vars.get("BENCHD_API_BASE", "https://openrouter.ai/api/v1")
    api_key = env_vars.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in cred file")
    return base_url, api_key


# ── Substring-bug-safe label parser ─────────────────────────────────
def parse_judge_label(raw_response: str) -> tuple[bool, str | None]:
    """Parse judge response with strict substring-bug protection.

    Official evaluate_qa.py does:
        label = 'yes' in eval_response.lower()
    This matches 'yesterday', 'eyes', 'bayesian', etc. — WRONG.

    Our approach (fail-closed):
    1. Normalize: strip, lowercase, remove trailing period
    2. Exact 'yes'  → True
    3. Exact 'no'   → False
    4. Starts with 'yes' + non-alpha → True, warning='ambiguous_startswith'
    5. Starts with 'no'  + non-alpha → False, warning='ambiguous_startswith'
    6. Contains 'yes' but not exact → False, warning='ambiguous_contains_yes'
    7. Contains 'no'  but not exact → False, warning='ambiguous_contains_no'
    8. Empty / unparseable           → False, warning='unparseable'
    9. API error / None              → False, warning='api_error'

    Returns (label: bool, warning: str | None).
    """
    if raw_response is None:
        return False, "api_error"

    normalized = raw_response.strip().lower()

    if not normalized:
        return False, "empty_response"

    # Remove trailing period (GPT-4o sometimes adds one)
    if normalized.endswith("."):
        normalized = normalized[:-1].strip()

    # Exact match
    if normalized == "yes":
        return True, None
    if normalized == "no":
        return False, None

    # Starts with yes/no followed by non-alpha (e.g., "yes." "yes\n" "yes,")
    if normalized.startswith("yes") and (
        len(normalized) == 3 or not normalized[3].isalpha()
    ):
        return True, "ambiguous_startswith"
    if normalized.startswith("no") and (
        len(normalized) == 2 or not normalized[2].isalpha()
    ):
        return False, "ambiguous_startswith"

    # Contains yes/no but not a valid prefix — fail-closed
    if "yes" in normalized:
        return False, "ambiguous_contains_yes"
    if "no" in normalized:
        return False, "ambiguous_contains_no"

    return False, "unparseable"


# ── OpenRouter client with backoff ───────────────────────────────────
@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APIError, openai.APIConnectionError),
    max_tries=5,
    max_time=120,
)
def judge_single(client: OpenAI, prompt: str) -> dict:
    """Send one judge call to OpenRouter GPT-4o. Returns raw result dict."""
    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        n=1,
        temperature=0,
        max_tokens=10,
    )
    choice = completion.choices[0]
    usage = completion.usage

    return {
        "raw_response": choice.message.content,
        "finish_reason": choice.finish_reason,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }


# ── Hashing helpers ──────────────────────────────────────────────────
def file_sha256(path: Path) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main evaluation loop ─────────────────────────────────────────────
def run_evaluation(
    hyp_file: Path,
    ref_file: Path,
    out_dir: Path,
    cred_file: Path | None = None,
) -> dict:
    """Run evaluation on all hypothesis items, return summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve credential file
    _cred_file = cred_file if cred_file is not None else CRED_FILE

    # Compute hashes
    hyp_hash = file_sha256(hyp_file)
    ref_hash = file_sha256(ref_file)

    # Load creds
    base_url, api_key = load_creds(_cred_file)
    print(f"[setup] OpenRouter base: {base_url}")
    print(f"[setup] Credentials loaded.")
    print(f"[setup] Judge model: {JUDGE_MODEL}")
    print(f"[setup] Official commit: {OFFICIAL_COMMIT}")

    # Build client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Load data
    hypotheses = [json.loads(line) for line in open(hyp_file).readlines()]
    references = json.load(open(ref_file))
    qid2qdata = {entry["question_id"]: entry for entry in references}
    qid2qtype = {entry["question_id"]: entry["question_type"] for entry in references}

    print(f"[data] {len(hypotheses)} hypotheses, {len(references)} references")

    # Run judge calls
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    errors = 0

    for i, entry in enumerate(hypotheses):
        qid = entry["question_id"]
        hyp = entry["hypothesis"]
        hyp_usage = entry.get("_usage", {})

        if qid not in qid2qtype:
            print(f"[warn] {qid}: not in reference data, skipping")
            continue

        qtype = qid2qtype[qid]
        question = qid2qdata[qid]["question"]
        answer = qid2qdata[qid]["answer"]
        is_abstention = "_abs" in qid

        # Build prompt using EXACT official function
        prompt = get_anscheck_prompt(
            qtype, question, answer, hyp, abstention=is_abstention
        )

        # Judge
        start = time.time()
        try:
            judge_result = judge_single(client, prompt)
        except Exception as e:
            judge_result = {
                "raw_response": None,
                "finish_reason": f"error: {e}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "error": str(e),
            }
            errors += 1

        latency = time.time() - start

        # Parse label with substring-bug protection
        label, warning = parse_judge_label(judge_result["raw_response"])

        # Cost
        prompt_tokens = judge_result.get("prompt_tokens", 0)
        completion_tokens = judge_result.get("completion_tokens", 0)
        cost_input = (prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT
        cost_output = (completion_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT
        item_cost = cost_input + cost_output

        total_input_tokens += prompt_tokens
        total_output_tokens += completion_tokens
        total_cost += item_cost

        # Build per-item record
        record = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "answer": answer,
            "hypothesis": hyp[:200] + "..." if len(hyp) > 200 else hyp,
            "autoeval_label": label,
            "raw_judge_response": judge_result["raw_response"],
            "finish_reason": judge_result.get("finish_reason"),
            "judge_model": JUDGE_MODEL,
            "official_commit": OFFICIAL_COMMIT,
            "hypothesis_file_hash": hyp_hash,
            "reference_file_hash": ref_hash,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "cost_usd": round(item_cost, 6),
            "latency_s": round(latency, 3),
            "parse_warning": warning,
            "hypothesis_usage": hyp_usage,
        }

        # Print per-item
        status = "✓" if label else "✗"
        warn_str = f" [{warning}]" if warning else ""
        print(
            f"[{i+1}/{len(hypotheses)}] {qid} {status}{warn_str} "
            f"({prompt_tokens}+{completion_tokens} tok, ${item_cost:.4f})"
        )

        results.append(record)

    # Compute summary
    n = len(results)
    correct = sum(1 for r in results if r["autoeval_label"])
    accuracy = correct / n if n > 0 else 0.0

    # By type
    type_breakdown = {}
    for r in results:
        qt = r["question_type"]
        if qt not in type_breakdown:
            type_breakdown[qt] = {"correct": 0, "total": 0}
        type_breakdown[qt]["total"] += 1
        if r["autoeval_label"]:
            type_breakdown[qt]["correct"] += 1

    summary = {
        "total_items": n,
        "correct": correct,
        "incorrect": n - correct,
        "errors": errors,
        "accuracy": round(accuracy, 4),
        "by_question_type": {
            qt: {
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0,
                "count": v["total"],
            }
            for qt, v in type_breakdown.items()
        },
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        },
        "total_cost_usd": round(total_cost, 6),
        "judge_model": JUDGE_MODEL,
        "official_commit": OFFICIAL_COMMIT,
        "hypothesis_file_hash": hyp_hash,
        "reference_file_hash": ref_hash,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = out_dir / f"eval-results-{timestamp}.json"
    summary_path = out_dir / f"eval-summary-{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[output] Results: {results_path}")
    print(f"[output] Summary: {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Items:        {n}")
    print(f"Correct:      {correct}")
    print(f"Incorrect:    {n - correct}")
    print(f"Errors:       {errors}")
    print(f"Accuracy:     {accuracy:.4f} ({correct}/{n})")
    print(f"Total cost:   ${total_cost:.6f}")
    print(f"Input tokens: {total_input_tokens}")
    print(f"Output tokens:{total_output_tokens}")
    print(f"\nBy question type:")
    for qt, info in summary["by_question_type"].items():
        print(f"  {qt}: {info['accuracy']:.4f} ({info['count']} items)")
    print(f"{'='*60}")

    return summary


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LongMemEval QA Evaluator — OpenRouter GPT-4o wrapper"
    )
    parser.add_argument(
        "--hyp-file",
        required=True,
        help="Path to hypothesis JSONL file",
    )
    parser.add_argument(
        "--ref-file",
        required=True,
        help="Path to reference JSON file (longmemeval_s_cleaned.json)",
    )
    parser.add_argument(
        "--out-dir",
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without making API calls",
    )
    parser.add_argument(
        "--cred-file",
        type=Path,
        default=CRED_FILE,
        help=f"Path to credential env file (default: {CRED_FILE})",
    )
    args = parser.parse_args()

    hyp_file = Path(args.hyp_file)
    ref_file = Path(args.ref_file)
    out_dir = Path(args.out_dir)

    if not hyp_file.exists():
        print(f"ERROR: Hypothesis file not found: {hyp_file}")
        sys.exit(1)
    if not ref_file.exists():
        print(f"ERROR: Reference file not found: {ref_file}")
        sys.exit(1)

    if args.dry_run:
        print("[dry-run] Validating setup...")
        base_url, api_key = load_creds(args.cred_file)
        print(f"[dry-run] OpenRouter base: {base_url}")
        print(f"[dry-run] Credentials loaded.")
        print(f"[dry-run] Judge model: {JUDGE_MODEL}")

        hypotheses = [
            json.loads(line) for line in open(hyp_file).readlines()
        ]
        references = json.load(open(ref_file))
        qid2qtype = {
            entry["question_id"]: entry["question_type"] for entry in references
        }
        print(f"[dry-run] Hypotheses: {len(hypotheses)} items")
        for i, hyp in enumerate(hypotheses):
            qid = hyp["question_id"]
            qtype = qid2qtype.get(qid, "UNKNOWN")
            is_abs = "_abs" in qid
            print(f"  [{i+1}] {qid} ({qtype}{', abstention' if is_abs else ''})")

        # Verify prompt function works
        q = references[0]["question"]
        a = references[0]["answer"]
        h = hypotheses[0]["hypothesis"]
        qt = qid2qtype[hypotheses[0]["question_id"]]
        prompt = get_anscheck_prompt(qt, q, a, h)
        print(f"\n[dry-run] Sample prompt length: {len(prompt)} chars")
        print(f"[dry-run] Sample prompt (first 300 chars):")
        print(f"  {prompt[:300]}...")
        print("\n[dry-run] Setup OK. Remove --dry-run to execute.")
        return

    run_evaluation(hyp_file, ref_file, out_dir, cred_file=args.cred_file)


if __name__ == "__main__":
    main()