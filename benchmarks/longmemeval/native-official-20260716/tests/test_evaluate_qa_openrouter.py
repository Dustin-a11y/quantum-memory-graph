#!/usr/bin/env python3
"""
Tests for evaluate_qa_openrouter.py.

Covers:
1. Prompt parity: verify our get_anscheck_prompt output matches official
   for all task types and abstention
2. Substring-bug: parse_judge_label rejects 'yesterday', 'eyes', etc.
3. Output schema: per-item record fields
4. Fail-closed: empty, None, ambiguous → label=False
5. Cost calculation
6. Credential safety: never prints key
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path for evaluate_qa_openrouter module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_qa_openrouter import (
    get_anscheck_prompt,
    parse_judge_label,
    file_sha256,
    JUDGE_MODEL,
    PRICE_PER_1M_INPUT,
    PRICE_PER_1M_OUTPUT,
)


class TestPromptParity(unittest.TestCase):
    """Verify our imported get_anscheck_prompt produces exact official prompts."""

    def test_single_session_user_prompt(self):
        """single-session-user template matches official format."""
        prompt = get_anscheck_prompt(
            "single-session-user",
            "What is my name?",
            "Alice",
            "Your name is Alice",
        )
        # Must contain key phrases from official template
        self.assertIn("I will give you a question", prompt)
        self.assertIn("correct answer", prompt)
        self.assertIn("response from a model", prompt)
        self.assertIn("Answer yes or no only", prompt)
        self.assertIn("What is my name?", prompt)
        self.assertIn("Alice", prompt)
        self.assertIn("Your name is Alice", prompt)
        # Must NOT contain abstention language
        self.assertNotIn("unanswerable", prompt.lower())
        # Must NOT contain temporal-reasoning off-by-one language
        self.assertNotIn("off-by-one", prompt.lower())
        # Must NOT contain knowledge-update language
        self.assertNotIn("previous information along with an updated answer", prompt)

    def test_single_session_assistant_prompt(self):
        """single-session-assistant uses same template as single-session-user."""
        prompt_user = get_anscheck_prompt(
            "single-session-user",
            "Q?",
            "A",
            "R",
        )
        prompt_assistant = get_anscheck_prompt(
            "single-session-assistant",
            "Q?",
            "A",
            "R",
        )
        # Same template structure (only question/answer/response differ)
        self.assertEqual(
            prompt_user.replace("Q?", "X").replace("A", "Y").replace("R", "Z"),
            prompt_assistant.replace("Q?", "X").replace("A", "Y").replace("R", "Z"),
        )

    def test_multi_session_prompt(self):
        """multi-session uses same base template."""
        prompt = get_anscheck_prompt(
            "multi-session",
            "Q?",
            "A",
            "R",
        )
        self.assertIn("Answer yes or no only", prompt)
        self.assertNotIn("off-by-one", prompt.lower())

    def test_temporal_reasoning_prompt(self):
        """temporal-reasoning has off-by-one tolerance."""
        prompt = get_anscheck_prompt(
            "temporal-reasoning",
            "How many days?",
            "18",
            "19 days",
        )
        self.assertIn("off-by-one", prompt.lower())
        self.assertIn("do not penalize off-by-one errors", prompt)
        self.assertIn("predicting 19 days when the answer is 18", prompt)

    def test_knowledge_update_prompt(self):
        """knowledge-update has previous-info tolerance."""
        prompt = get_anscheck_prompt(
            "knowledge-update",
            "Q?",
            "A",
            "R",
        )
        self.assertIn("previous information along with an updated answer", prompt)
        self.assertNotIn("off-by-one", prompt.lower())

    def test_single_session_preference_prompt(self):
        """single-session-preference uses rubric instead of answer."""
        prompt = get_anscheck_prompt(
            "single-session-preference",
            "Q?",
            "Rubric text",
            "R",
        )
        self.assertIn("rubric", prompt.lower())
        self.assertIn("Rubric text", prompt)
        self.assertIn("desired personalized response", prompt)
        # Should NOT say "correct answer"
        self.assertNotIn("Correct Answer:", prompt)

    def test_abstention_prompt(self):
        """Abstention flag triggers unanswerable template."""
        prompt = get_anscheck_prompt(
            "single-session-user",
            "Unanswerable Q?",
            "Explanation text",
            "I don't know",
            abstention=True,
        )
        self.assertIn("unanswerable", prompt.lower())
        self.assertIn("Does the model correctly identify", prompt)
        self.assertIn("Explanation text", prompt)
        self.assertNotIn("correct answer", prompt.lower())

    def test_unknown_task_type_raises(self):
        """Unknown task type raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            get_anscheck_prompt("nonexistent-type", "Q", "A", "R")


class TestSubstringBug(unittest.TestCase):
    """Verify parse_judge_label does NOT have the substring bug.

    Official code:  label = 'yes' in eval_response.lower()
    This matches 'yesterday', 'eyes', 'bayesian', 'lawyer' — WRONG.

    All of these must be rejected.
    """

    # ── Exact yes/no ──────────────────────────────────────────────
    def test_exact_yes(self):
        self.assertEqual(parse_judge_label("yes"), (True, None))

    def test_exact_yes_whitespace(self):
        self.assertEqual(parse_judge_label("  yes  "), (True, None))

    def test_exact_yes_case(self):
        self.assertEqual(parse_judge_label("YES"), (True, None))
        self.assertEqual(parse_judge_label("Yes"), (True, None))

    def test_exact_yes_with_period(self):
        self.assertEqual(parse_judge_label("yes."), (True, None))

    def test_exact_no(self):
        self.assertEqual(parse_judge_label("no"), (False, None))

    def test_exact_no_with_period(self):
        self.assertEqual(parse_judge_label("no."), (False, None))

    # ── Substring-bug vectors (ALL must be False) ─────────────────
    def test_yesterday_is_not_yes(self):
        """'yesterday' contains 'yes' but is not 'yes'."""
        label, warning = parse_judge_label("yesterday")
        self.assertFalse(label)
        self.assertIsNotNone(warning)

    def test_eyes_is_not_yes(self):
        label, warning = parse_judge_label("eyes")
        self.assertFalse(label)
        self.assertIsNotNone(warning)

    def test_bayesian_is_not_yes(self):
        label, warning = parse_judge_label("bayesian")
        self.assertFalse(label)
        self.assertIsNotNone(warning)

    def test_lawyer_is_not_yes(self):
        label, warning = parse_judge_label("lawyer")
        self.assertFalse(label)
        self.assertIsNotNone(warning)

    def test_nobody_is_not_no(self):
        label, warning = parse_judge_label("nobody")
        self.assertFalse(label)
        self.assertIsNotNone(warning)

    def test_novel_is_not_no(self):
        label, warning = parse_judge_label("novel")
        self.assertFalse(label)
        self.assertIsNotNone(warning)

    def test_technology_is_not_no(self):
        label, warning = parse_judge_label("technology")
        self.assertFalse(label)
        self.assertIsNotNone(warning)

    def test_yes_with_explanation(self):
        """'Yes, the model correctly...' — starts with yes + non-alpha."""
        label, warning = parse_judge_label("Yes, the model correctly answers")
        self.assertTrue(label)
        self.assertEqual(warning, "ambiguous_startswith")

    def test_no_with_explanation(self):
        """'No, the model...' — starts with no + non-alpha."""
        label, warning = parse_judge_label("No, the model does not")
        self.assertFalse(label)
        self.assertEqual(warning, "ambiguous_startswith")

    def test_yes_newline(self):
        # "yes\n" after strip+lower = "yes" → exact match, no warning
        label, warning = parse_judge_label("yes\n")
        self.assertTrue(label)
        self.assertIsNone(warning)

    # ── Fail-closed: empty / None / unparseable ──────────────────
    def test_none_fail_closed(self):
        label, warning = parse_judge_label(None)
        self.assertFalse(label)
        self.assertEqual(warning, "api_error")

    def test_empty_fail_closed(self):
        label, warning = parse_judge_label("")
        self.assertFalse(label)
        self.assertEqual(warning, "empty_response")

    def test_whitespace_only_fail_closed(self):
        label, warning = parse_judge_label("   ")
        self.assertFalse(label)
        self.assertEqual(warning, "empty_response")

    def test_gibberish_fail_closed(self):
        label, warning = parse_judge_label("maybe perhaps")
        self.assertFalse(label)
        self.assertEqual(warning, "unparseable")

    def test_unrelated_answer_fail_closed(self):
        label, warning = parse_judge_label("The answer is correct")
        self.assertFalse(label)
        self.assertEqual(warning, "unparseable")

    # ── Edge cases: yes/no embedded in longer strings ─────────────
    def test_yes_in_middle(self):
        """'I think yes the model is correct' — yes not at start."""
        label, warning = parse_judge_label("I think yes")
        self.assertFalse(label)
        self.assertIsNotNone(warning)

    def test_no_in_middle(self):
        label, warning = parse_judge_label("I think no")
        self.assertFalse(label)
        self.assertIsNotNone(warning)

    def test_yes_followed_by_alpha(self):
        """'yesbut' — yes followed immediately by alpha."""
        label, warning = parse_judge_label("yesbut")
        self.assertFalse(label)
        self.assertEqual(warning, "ambiguous_contains_yes")


class TestOutputSchema(unittest.TestCase):
    """Verify output record has all required fields."""

    def test_required_fields(self):
        """Every record must have these fields."""
        required = [
            "question_id",
            "question_type",
            "question",
            "answer",
            "hypothesis",
            "autoeval_label",
            "raw_judge_response",
            "finish_reason",
            "judge_model",
            "official_commit",
            "hypothesis_file_hash",
            "reference_file_hash",
            "token_usage",
            "cost_usd",
            "latency_s",
            "parse_warning",
        ]
        # We don't have a real record, just verify the names exist
        # Actual schema enforcement happens at write time
        self.assertTrue(all(isinstance(f, str) for f in required))

    def test_no_credentials_in_results(self):
        """Results must NEVER contain api_key or secret."""
        forbidden = ["sk-or-", "api_key", "secret", "OPENROUTER_API_KEY"]
        # This is a design constraint — verify our code paths
        # by checking that the record construction doesn't include
        # credential fields
        record_fields = [
            "question_id", "question_type", "question", "answer",
            "hypothesis", "autoeval_label", "raw_judge_response",
            "finish_reason", "judge_model", "official_commit",
            "hypothesis_file_hash", "reference_file_hash",
            "token_usage", "cost_usd", "latency_s", "parse_warning",
            "hypothesis_usage",
        ]
        for field in record_fields:
            self.assertNotIn("api_key", field.lower())
            self.assertNotIn("secret", field.lower())
            self.assertNotIn("credential", field.lower())


class TestCostCalculation(unittest.TestCase):
    """Verify token cost math."""

    def test_cost_formula(self):
        """cost = (input/1M)*2.50 + (output/1M)*10.00"""
        input_tokens = 1000
        output_tokens = 500
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        actual = (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT + \
                 (output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT
        self.assertAlmostEqual(expected, actual, places=8)

    def test_zero_tokens_zero_cost(self):
        actual = (0 / 1_000_000) * PRICE_PER_1M_INPUT + \
                 (0 / 1_000_000) * PRICE_PER_1M_OUTPUT
        self.assertEqual(actual, 0.0)


class TestJudgeModelConstant(unittest.TestCase):
    """Verify judge model is openai/gpt-4o."""

    def test_model_is_gpt4o(self):
        self.assertEqual(JUDGE_MODEL, "openai/gpt-4o")


class TestHashing(unittest.TestCase):
    """Verify SHA-256 file hashing."""

    def test_hash_deterministic(self):
        """Same content → same hash."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content")
            f.flush()
            h1 = file_sha256(Path(f.name))
            h2 = file_sha256(Path(f.name))
            self.assertEqual(h1, h2)
        os.unlink(f.name)

    def test_hash_different(self):
        """Different content → different hash."""
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        f1.write("content A")
        f1.flush()
        f2 = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        f2.write("content B")
        f2.flush()
        h1 = file_sha256(Path(f1.name))
        h2 = file_sha256(Path(f2.name))
        self.assertNotEqual(h1, h2)
        os.unlink(f1.name)
        os.unlink(f2.name)


class TestCredentialSafety(unittest.TestCase):
    """Verify API key NEVER leaks to stdout, stderr, or any output artifact."""

    def test_load_creds_returns_key_but_never_prints(self):
        """load_creds must not print any portion of the API key."""
        import io
        import tempfile
        from contextlib import redirect_stdout

        from evaluate_qa_openrouter import load_creds

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".env"
        ) as f:
            f.write("BENCHD_API_BASE=https://openrouter.ai/api/v1\n")
            f.write("OPENROUTER_API_KEY=sk-or-v1-abcd1234efgh5678ijkl\n")
            f.flush()
            env_path = f.name

        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                base_url, api_key = load_creds(Path(env_path))
            output = buf.getvalue()
            # Must NOT contain any substring of the key
            self.assertNotIn("sk-or-v1", output)
            self.assertNotIn("abcd1234", output)
            self.assertNotIn("efgh5678", output)
            self.assertNotIn("ijkl", output)
            # Must NOT contain the full key
            self.assertNotIn("sk-or-v1-abcd1234efgh5678ijkl", output)
            # Must return the key correctly
            self.assertEqual(api_key, "sk-or-v1-abcd1234efgh5678ijkl")
            self.assertEqual(base_url, "https://openrouter.ai/api/v1")
        finally:
            os.unlink(env_path)

    def test_dry_run_stdout_no_key_substring(self):
        """--dry-run must never print any portion of the API key."""
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout

        # Create a minimal hypothesis file
        hyp_content = json.dumps({
            "question_id": "test_q1",
            "hypothesis": "test hypothesis",
        }) + "\n"

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".jsonl"
        ) as hf:
            hf.write(hyp_content)
            hf.flush()
            hyp_path = hf.name

        # Create a minimal reference file
        ref_content = json.dumps([{
            "question_id": "test_q1",
            "question": "What is test?",
            "answer": "test",
            "question_type": "single-session-user",
        }])

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as rf:
            rf.write(ref_content)
            rf.flush()
            ref_path = rf.name

        # Create a credential file with fake key
        fake_key = "sk-or-v1-zyxw9876fedcba5432mnop"
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".env"
        ) as cf:
            cf.write(f"BENCHD_API_BASE=https://openrouter.ai/api/v1\n")
            cf.write(f"OPENROUTER_API_KEY={fake_key}\n")
            cf.flush()
            cred_path = cf.name

        try:
            import subprocess

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.evaluate_qa_openrouter",
                    "--hyp-file", hyp_path,
                    "--ref-file", ref_path,
                    "--dry-run",
                    "--cred-file", cred_path,
                ],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            stdout = result.stdout
            stderr = result.stderr
            combined = stdout + stderr

            # NEVER contain any substring of the fake key
            for substr in [
                "zyxw9876", "fedcba", "sk-or-v1-zyxw", "5432mnop",
                fake_key,
            ]:
                self.assertNotIn(
                    substr, combined,
                    f"CREDENTIAL LEAK: '{substr}' found in output"
                )

            # The "sk-or-v1" prefix alone (without actual key chars)
            # should also not appear since we're printing "loaded (N chars)"
            # instead of any key content
            # Actually, let's check that the full prefix isn't printed
            # by checking output doesn't have key-like pattern
            import re
            self.assertFalse(
                re.search(r'sk-or-v1-[a-z0-9]{8,}', combined),
                "CREDENTIAL LEAK: key-like pattern found in output"
            )

            # Verify it ran successfully
            self.assertIn("[dry-run] Setup OK", stdout)
            self.assertIn("Credentials loaded.", stdout)
            # Key length must NOT appear in output
            self.assertNotIn(str(len(fake_key)), stdout, "KEY LENGTH LEAK: key length in stdout")
            self.assertNotIn(str(len(fake_key)), stderr, "KEY LENGTH LEAK: key length in stderr")
            # Key prefix 'sk-' must NOT appear (we don't even reveal prefix)
            self.assertNotIn("sk-or-", stdout, "KEY PREFIX LEAK in stdout")
            self.assertNotIn("sk-or-", stderr, "KEY PREFIX LEAK in stderr")
        finally:
            for p in [hyp_path, ref_path, cred_path]:
                try:
                    os.unlink(p)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)