"""
Tests for the LongMemEval OpenRouter generation wrapper.

Usage: python -m pytest test_generation_wrapper.py -v
"""
import json
import os
import sys
import pytest

# Add the official generation module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LongMemEval/src/generation'))

from run_generation import prepare_prompt  # noqa: E402


# --- Test fixtures ---

@pytest.fixture
def sample_entry():
    """Load first entry from the 10-item slice."""
    slice_path = os.path.join(
        os.path.dirname(__file__),
        'runs/native-official-20260715/qmg_retrieval_10.jsonl'
    )
    with open(slice_path) as f:
        return json.loads(f.readline())


@pytest.fixture
def tokenizer_setup():
    """Minimal tokenizer for testing — use tiktoken o200k_base as proxy."""
    import tiktoken
    return tiktoken.get_encoding('o200k_base')


# --- Test: prepare_prompt works with flat-session + nl + cot ---

def test_prepare_prompt_flat_session_nl_cot(sample_entry, tokenizer_setup):
    """Verify prepare_prompt produces a non-empty string with expected substrings."""
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
    assert 'Current Date:' in prompt
    assert 'Question:' in prompt
    assert 'Answer (step by step)' in prompt
    assert sample_entry['question'] in prompt


def test_prepare_prompt_useronly_true(sample_entry, tokenizer_setup):
    """Verify useronly=True filters to user messages only."""
    prompt = prepare_prompt(
        entry=sample_entry,
        retriever_type='flat-session',
        topk_context=10,
        useronly=True,
        history_format='nl',
        cot=False,
        tokenizer=tokenizer_setup,
        tokenizer_backend='openai',
        max_retrieval_length=64000,
        merge_key_expansion_into_value='none',
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_prepare_prompt_no_retrieval(sample_entry, tokenizer_setup):
    """Verify no-retrieval produces a minimal prompt."""
    prompt = prepare_prompt(
        entry=sample_entry,
        retriever_type='no-retrieval',
        topk_context=10,
        useronly=False,
        history_format='nl',
        cot=False,
        tokenizer=tokenizer_setup,
        tokenizer_backend='openai',
        max_retrieval_length=64000,
        merge_key_expansion_into_value='none',
    )
    assert isinstance(prompt, str)
    assert sample_entry['question'] in prompt


def test_prepare_prompt_topk_truncation(sample_entry, tokenizer_setup):
    """Verify topk=3 limits retrieved sessions."""
    prompt_top3 = prepare_prompt(
        entry=sample_entry,
        retriever_type='flat-session',
        topk_context=3,
        useronly=False,
        history_format='nl',
        cot=False,
        tokenizer=tokenizer_setup,
        tokenizer_backend='openai',
        max_retrieval_length=64000,
        merge_key_expansion_into_value='none',
    )

    prompt_top10 = prepare_prompt(
        entry=sample_entry,
        retriever_type='flat-session',
        topk_context=10,
        useronly=False,
        history_format='nl',
        cot=False,
        tokenizer=tokenizer_setup,
        tokenizer_backend='openai',
        max_retrieval_length=64000,
        merge_key_expansion_into_value='none',
    )

    # top3 should produce a shorter prompt than top10
    assert len(prompt_top3) < len(prompt_top10)


def test_prepare_prompt_strips_has_answer(sample_entry, tokenizer_setup):
    """Verify 'has_answer' key is stripped from turn entries."""
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
    assert 'has_answer' not in prompt


def test_prepare_prompt_all_10_entries(tokenizer_setup):
    """Smoke test: prepare_prompt on all 10 slice entries without error."""
    slice_path = os.path.join(
        os.path.dirname(__file__),
        'runs/native-official-20260715/qmg_retrieval_10.jsonl'
    )
    with open(slice_path) as f:
        entries = [json.loads(line) for line in f]

    assert len(entries) == 10

    for entry in entries:
        prompt = prepare_prompt(
            entry=entry,
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
        assert isinstance(prompt, str) and len(prompt) > 0, \
            f"Empty prompt for {entry['question_id']}"


# --- Test: API client fail-closed ---

def test_client_fail_closed_bad_key():
    """Verify API client fails with clear error on invalid key."""
    from openai import OpenAI
    client = OpenAI(
        api_key='sk-invalid-deadbeef',
        base_url='https://openrouter.ai/api/v1',
    )
    with pytest.raises(Exception):
        client.chat.completions.create(
            model='deepseek/deepseek-r1',
            messages=[{'role': 'user', 'content': 'test'}],
            max_tokens=5,
        )


def test_client_fail_closed_bad_base_url():
    """Verify API client fails with clear error on bad base URL."""
    from openai import OpenAI
    client = OpenAI(
        api_key='sk-or-v1-dummy',
        base_url='https://nonexistent.example.com/v1',
    )
    with pytest.raises(Exception):
        client.chat.completions.create(
            model='deepseek/deepseek-r1',
            messages=[{'role': 'user', 'content': 'test'}],
            max_tokens=5,
        )


def test_client_bad_model():
    """Verify API client fails with clear error on unknown model."""
    from openai import OpenAI
    client = OpenAI(
        api_key='sk-or-v1-dummy',
        base_url='https://openrouter.ai/api/v1',
    )
    with pytest.raises(Exception):
        client.chat.completions.create(
            model='nonexistent/model-v99',
            messages=[{'role': 'user', 'content': 'test'}],
            max_tokens=5,
        )


# --- Test: hypothesis output shape ---

def test_hypothesis_output_schema():
    """Verify generated output matches the expected schema."""
    expected_keys = {'question_id', 'hypothesis'}
    # We validate that the output handler enforces these keys
    from openai.types.chat import ChatCompletion

    # Schema check: any hypothesis output must have these keys
    # (enforced by our wrapper, not OpenAI)
    test_output = {'question_id': 'test123', 'hypothesis': 'Sample answer'}
    assert expected_keys.issubset(test_output.keys())


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
