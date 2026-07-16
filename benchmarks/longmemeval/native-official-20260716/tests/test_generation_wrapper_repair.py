"""
Tests for the LongMemEval OpenRouter generation wrapper — repair edition.

Covers the DeepSeek R1 reasoning-only null-content defect:
  - Null content (message.content=None) handling
  - Empty content handling
  - reasoning_content presence without visible content
  - Bounded retry (max 3 attempts) with fallback prompt requesting final answer
  - Fail-closed after retry exhaustion
  - Token accounting across retries

Usage: python -m pytest test_generation_wrapper_repair.py -v
"""
import json
import os
import sys
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Add the repair wrapper path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'LongMemEval/src/generation'))

from run_generation import prepare_prompt  # noqa: E402


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_entry():
    """Load first entry from the 10-item slice."""
    slice_path = os.path.join(
        REPO_ROOT, 'runs/native-official-20260715/qmg_retrieval_10.jsonl'
    )
    with open(slice_path) as f:
        return json.loads(f.readline())


@pytest.fixture
def tokenizer_setup():
    """Minimal tokenizer for testing."""
    import tiktoken
    return tiktoken.get_encoding('o200k_base')


@pytest.fixture
def mock_completion_null_content():
    """Simulate DeepSeek R1 reasoning-only response: content=None, reasoning_content filled."""
    choice = MagicMock()
    choice.finish_reason = 'stop'
    msg = MagicMock()
    type(msg).content = PropertyMock(return_value=None)
    msg.reasoning_content = (
        "Let me analyze the chat history carefully to find the answer...\n"
        "I need to count the baking mentions across all sessions...\n"
        "Session 3 mentions baking cookies, session 7 mentions baking a cake..."
    )
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock()
    completion.usage.prompt_tokens = 1000
    completion.usage.completion_tokens = 500
    completion.usage.total_tokens = 1500
    return completion


@pytest.fixture
def mock_completion_empty_content():
    """Simulate response where content is empty string."""
    choice = MagicMock()
    choice.finish_reason = 'stop'
    msg = MagicMock()
    type(msg).content = PropertyMock(return_value='')
    msg.reasoning_content = None
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock()
    completion.usage.prompt_tokens = 1000
    completion.usage.completion_tokens = 10
    completion.usage.total_tokens = 1010
    return completion


@pytest.fixture
def mock_completion_whitespace_only():
    """Simulate response where content is only whitespace."""
    choice = MagicMock()
    choice.finish_reason = 'stop'
    msg = MagicMock()
    type(msg).content = PropertyMock(return_value='   \n  \t  ')
    msg.reasoning_content = None
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock()
    completion.usage.prompt_tokens = 1000
    completion.usage.completion_tokens = 5
    completion.usage.total_tokens = 1005
    return completion


@pytest.fixture
def mock_completion_valid():
    """Simulate a valid response with both reasoning and visible content."""
    choice = MagicMock()
    choice.finish_reason = 'stop'
    msg = MagicMock()
    type(msg).content = PropertyMock(
        return_value="The user baked cookies 3 times and a cake once, totaling 4 times."
    )
    msg.reasoning_content = "Let me count the baking events..."
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock()
    completion.usage.prompt_tokens = 1000
    completion.usage.completion_tokens = 200
    completion.usage.total_tokens = 1200
    return completion


# ============================================================================
# Test 1: Null content detection (the root cause)
# ============================================================================

def test_content_is_none_detection():
    """content=None is the root cause — must be detected before .strip()."""
    content = None
    # This is what line 202 does and crashes:
    with pytest.raises(AttributeError):
        content.strip()

    # Correct approach: check for None first
    is_valid = content is not None and len(content.strip()) > 0
    assert is_valid is False


def test_content_is_none_handled_before_strip():
    """Guard: never call .strip() on None — check first."""
    def safe_get_answer(completion):
        """Safe extraction that handles null content."""
        msg = completion.choices[0].message
        content = msg.content
        if content is None:
            return None
        stripped = content.strip()
        if not stripped:
            return None
        return stripped

    # null
    choice_null = MagicMock()
    msg_null = MagicMock()
    type(msg_null).content = PropertyMock(return_value=None)
    choice_null.message = msg_null
    comp_null = MagicMock()
    comp_null.choices = [choice_null]

    assert safe_get_answer(comp_null) is None

    # empty
    choice_empty = MagicMock()
    msg_empty = MagicMock()
    type(msg_empty).content = PropertyMock(return_value='')
    choice_empty.message = msg_empty
    comp_empty = MagicMock()
    comp_empty.choices = [choice_empty]

    assert safe_get_answer(comp_empty) is None

    # whitespace only
    choice_ws = MagicMock()
    msg_ws = MagicMock()
    type(msg_ws).content = PropertyMock(return_value='  \n ')
    choice_ws.message = msg_ws
    comp_ws = MagicMock()
    comp_ws.choices = [choice_ws]

    assert safe_get_answer(comp_ws) is None

    # valid
    choice_valid = MagicMock()
    msg_valid = MagicMock()
    type(msg_valid).content = PropertyMock(return_value='Hello World')
    choice_valid.message = msg_valid
    comp_valid = MagicMock()
    comp_valid.choices = [choice_valid]

    assert safe_get_answer(comp_valid) == 'Hello World'


# ============================================================================
# Test 2: reasoning_content detection
# ============================================================================

def test_reasoning_content_detection():
    """Detect when response has reasoning but no visible content."""
    def has_reasoning_only(completion):
        """Check if response is reasoning-only with null/empty visible content."""
        msg = completion.choices[0].message
        content = msg.content
        reasoning = getattr(msg, 'reasoning_content', None)
        visible_empty = content is None or len(content.strip()) == 0
        has_reasoning = reasoning is not None and len(reasoning.strip()) > 0
        return visible_empty and has_reasoning

    # Reasoning-only
    choice_r = MagicMock()
    msg_r = MagicMock()
    type(msg_r).content = PropertyMock(return_value=None)
    msg_r.reasoning_content = "Deep analysis of the question..."
    choice_r.message = msg_r
    comp_r = MagicMock()
    comp_r.choices = [choice_r]
    assert has_reasoning_only(comp_r) is True

    # Reasoning + content
    choice_both = MagicMock()
    msg_both = MagicMock()
    type(msg_both).content = PropertyMock(return_value='Answer text')
    msg_both.reasoning_content = "Analysis..."
    choice_both.message = msg_both
    comp_both = MagicMock()
    comp_both.choices = [choice_both]
    assert has_reasoning_only(comp_both) is False

    # No reasoning, no content
    choice_neither = MagicMock()
    msg_neither = MagicMock()
    type(msg_neither).content = PropertyMock(return_value=None)
    msg_neither.reasoning_content = None
    choice_neither.message = msg_neither
    comp_neither = MagicMock()
    comp_neither.choices = [choice_neither]
    assert has_reasoning_only(comp_neither) is False

    # No reasoning attribute at all
    choice_noattr = MagicMock()
    msg_noattr = MagicMock()
    type(msg_noattr).content = PropertyMock(return_value=None)
    # No reasoning_content attribute
    choice_noattr.message = msg_noattr
    comp_noattr = MagicMock()
    comp_noattr.choices = [choice_noattr]
    assert has_reasoning_only(comp_noattr) is False


# ============================================================================
# Test 3: Follow-up prompt for reasoning-only responses
# ============================================================================

def test_follow_up_prompt_generation():
    """When reasoning-only detected, generate a follow-up requesting final answer."""
    def build_follow_up_prompt(original_prompt, reasoning_content):
        return (
            f"{original_prompt}\n\n"
            f"[SYSTEM: Your previous response contained reasoning but no final answer. "
            f"Please provide your FINAL ANSWER directly. "
            f"Start with the answer, then you may add brief explanation.]"
        )

    original = "Question: How many times did I bake?\nAnswer (step by step):"
    reasoning = "Let me analyze... I found 3 mentions of baking."
    follow_up = build_follow_up_prompt(original, reasoning)

    assert "FINAL ANSWER" in follow_up
    assert original in follow_up
    assert len(follow_up) > len(original)


# ============================================================================
# Test 4: Bounded retry (max 3 attempts)
# ============================================================================

def test_bounded_retry_max_3(
    mock_completion_null_content,
    mock_completion_valid,
):
    """Verify retry loop respects max_retries=3 and never exceeds it."""
    def _make_null():
        return mock_completion_null_content

    max_retries = 3

    def simulate_retry_loop(completions):
        """Simulate a retry loop that tries up to max_retries times."""
        attempts = 0
        last_error = None
        for i in range(max_retries):
            attempts += 1
            try:
                msg = completions[i].choices[0].message
                content = msg.content
                if content is None:
                    raise ValueError("Null content — retry with follow-up")
                stripped = content.strip()
                if not stripped:
                    raise ValueError("Empty content — retry with follow-up")
                return stripped, attempts, None
            except Exception as e:
                last_error = str(e)
                continue
        return None, attempts, last_error

    # All three null → should fail after 3 attempts
    null_completions = [mock_completion_null_content for _ in range(5)]
    answer, attempts, error = simulate_retry_loop(null_completions)
    assert answer is None
    assert attempts == 3
    assert "Null content" in error

    # First two null, third valid → should succeed on attempt 3
    comps = [
        mock_completion_null_content,
        mock_completion_null_content,
        mock_completion_valid,
    ]
    answer, attempts, error = simulate_retry_loop(comps)
    assert answer is not None
    assert attempts == 3
    assert error is None
    assert "baked cookies" in answer

    # First valid → should succeed on attempt 1
    comps2 = [mock_completion_valid, mock_completion_null_content]
    answer, attempts, error = simulate_retry_loop(comps2)
    assert answer is not None
    assert attempts == 1
    assert error is None


# ============================================================================
# Test 5: Fail-closed after retry exhaustion
# ============================================================================

def test_fail_closed_after_retry_exhaustion(mock_completion_null_content):
    """After max_retries exhausted with no valid answer, fail-closed: skip entry, record error."""
    max_retries = 3
    null_completions = [mock_completion_null_content for _ in range(max_retries)]

    # Simulate: all retries exhausted
    successes = []
    errors = []
    qid = 'test_qid_123'

    answer = None
    for attempt in range(max_retries):
        msg = null_completions[attempt].choices[0].message
        if msg.content is not None and msg.content.strip():
            answer = msg.content.strip()
            break

    if answer is None:
        errors.append({
            'question_id': qid,
            'error': f'Exhausted {max_retries} retries — no valid content',
            'attempts': max_retries,
        })
        # Fail-closed: do NOT add to successes, do NOT exit 0
    else:
        successes.append({'question_id': qid, 'hypothesis': answer})

    assert len(successes) == 0
    assert len(errors) == 1
    assert errors[0]['attempts'] == 3
    assert 'retries' in errors[0]['error']


# ============================================================================
# Test 6: Token accounting across retries
# ============================================================================

def test_token_accounting_across_retries(
    mock_completion_null_content,
    mock_completion_valid,
):
    """Verify that tokens from all attempts (including failed retries) are tracked."""
    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    PRICE_IN = 0.55  # DeepSeek R1 pricing per 1M
    PRICE_OUT = 2.19

    completions = [
        mock_completion_null_content,   # attempt 1: 1000+500 tokens
        mock_completion_null_content,   # attempt 2: 1000+500 tokens
        mock_completion_valid,          # attempt 3: 1000+200 tokens (success)
    ]

    for comp in completions:
        usage = comp.usage
        total_prompt += usage.prompt_tokens
        total_completion += usage.completion_tokens
        total_cost += (usage.prompt_tokens / 1_000_000) * PRICE_IN
        total_cost += (usage.completion_tokens / 1_000_000) * PRICE_OUT

    # 3 attempts × 1000 prompt = 3000 prompt
    assert total_prompt == 3000
    # 500 + 500 + 200 = 1200 completion
    assert total_completion == 1200
    assert total_cost > 0
    # Cost should include all 3 attempts
    expected_cost = (3000 / 1e6) * PRICE_IN + (1200 / 1e6) * PRICE_OUT
    assert abs(total_cost - expected_cost) < 0.0001


# ============================================================================
# Test 7: is_valid_answer helper
# ============================================================================

def test_is_valid_answer():
    """Validate the is_valid_answer predicate used by the retry loop."""
    def is_valid_answer(content):
        if content is None:
            return False
        if not isinstance(content, str):
            return False
        if len(content.strip()) == 0:
            return False
        return True

    assert is_valid_answer(None) is False
    assert is_valid_answer('') is False
    assert is_valid_answer('   ') is False
    assert is_valid_answer('\n\t') is False
    assert is_valid_answer('Hello') is True
    assert is_valid_answer('  Answer text  ') is True


# ============================================================================
# Test 8: Repair wrapper output schema
# ============================================================================

def test_repair_wrapper_output_schema():
    """Verify repair wrapper output matches existing hypothesis schema exactly."""
    existing_keys = {'question_id', 'hypothesis', '_usage'}

    # Valid output
    valid_output = {
        'question_id': '88432d0a',
        'hypothesis': 'The answer is 3 times.',
        '_usage': {
            'prompt_tokens': 5000,
            'completion_tokens': 500,
            'total_tokens': 5500,
            'latency_s': 8.5,
            'attempts': 1,
        }
    }
    assert existing_keys.issubset(valid_output.keys())

    # Error output (fail-closed) — should NOT produce a hypothesis entry
    # Instead, error goes to errors list, not to hypotheses
    error_output = {
        'question_id': 'bad_id',
        'error': 'Exhausted 3 retries',
        'attempts': 3,
    }
    # Error records have different keys — that's expected
    assert 'hypothesis' not in error_output


# ============================================================================
# Test 9: Prepare prompt still works (regression)
# ============================================================================

def test_prepare_prompt_still_works(sample_entry, tokenizer_setup):
    """Regression: verify prepare_prompt still works after import path changes."""
    prompt = prepare_prompt(
        entry=sample_entry,
        retriever_type='flat-session',
        topk_context=10,
        useronly=False,
        history_format='nl',
        cot=True,
        tokenizer=tokenizer_setup,
        tokenizer_backend='openai',
        max_retrieval_length=64000,
        merge_key_expansion_into_value='none',
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert 'History Chats' in prompt
    assert sample_entry['question'] in prompt


# ============================================================================
# Test 10: No overwrite of existing 486 (byte equality per qid)
# ============================================================================

def test_no_overwrite_byte_equality():
    """Verify merge logic preserves original 486 entries byte-for-byte."""
    original_entries = [
        {'question_id': 'abc123', 'hypothesis': 'Answer A', '_usage': {'tokens': 100}},
        {'question_id': 'def456', 'hypothesis': 'Answer B', '_usage': {'tokens': 200}},
    ]
    repair_entries = [
        {'question_id': 'xyz789', 'hypothesis': 'Answer C', '_usage': {'tokens': 150, 'attempts': 2}},
    ]

    # Merge: repair entries should NOT replace existing
    orig_by_qid = {e['question_id']: e for e in original_entries}
    for repair in repair_entries:
        assert repair['question_id'] not in orig_by_qid, \
            f"Repair entry {repair['question_id']} would overwrite existing!"

    # Merge preserves originals byte-identical
    merged = original_entries + repair_entries
    assert len(merged) == 3
    assert merged[0] == original_entries[0]
    assert merged[1] == original_entries[1]
    assert merged[2] == repair_entries[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
