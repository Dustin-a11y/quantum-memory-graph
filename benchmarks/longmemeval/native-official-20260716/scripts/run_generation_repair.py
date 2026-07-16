#!/usr/bin/env python3
"""
LongMemEval Generation Repair Wrapper — bounded-retry for DeepSeek R1 null-content.

Repairs items where DeepSeek R1 returned reasoning_content but content=None
(original wrapper line 202 calls .strip() on None → AttributeError → skipped).

This wrapper:
  - Detects null/empty/whitespace-only content
  - On reasoning-only: builds explicit follow-up requesting final visible answer
  - Bounded retry: max 3 API calls per item
  - Accounts tokens across all attempts (including failed retries)
  - Fail-closed: if all 3 retries fail, records error, skips entry, exits nonzero
  - Outputs JSONL with exact same schema as original: {question_id, hypothesis, _usage{..., attempts}}

Usage:
    python run_generation_repair.py \
        --in-file runs/native-official-20260715/qmg_retrieval_14_missing.jsonl \
        --out-dir runs/native-official-20260715 \
        --cred-file ./benchd-openrouter.env

Credentials: loaded from --cred-file at runtime, NEVER stored or printed.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import tiktoken
from openai import OpenAI

# Add official generation module to path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, 'LongMemEval/src/generation'))
from run_generation import prepare_prompt  # noqa: E402


# ---------------------------------------------------------------------------
# DeepSeek R1 pricing (OpenRouter, July 2025)
# ---------------------------------------------------------------------------
PRICE_PER_1M_INPUT = 0.55   # $0.55/M input tokens
PRICE_PER_1M_OUTPUT = 2.19  # $2.19/M output tokens
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='LongMemEval generation repair — bounded retry for DeepSeek R1 null-content'
    )
    parser.add_argument('--in-file', required=True, help='JSONL retrieval slice (14 missing items)')
    parser.add_argument('--out-dir', required=True, help='Output directory for repaired hypotheses')
    parser.add_argument('--cred-file', required=True, help='.env file with OPENROUTER_API_KEY, BENCHD_API_BASE')
    parser.add_argument('--model', default='deepseek/deepseek-r1', help='OpenRouter model ID')
    parser.add_argument('--max-context', type=int, default=64000)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--gen-length', type=int, default=800)
    parser.add_argument('--retriever-type', default='flat-session')
    parser.add_argument('--history-format', default='nl')
    parser.add_argument('--useronly', action='store_true', default=False)
    parser.add_argument('--cot', action='store_true', default=True)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def load_credentials(cred_file: str) -> dict:
    if not os.path.isfile(cred_file):
        raise FileNotFoundError(f'Credential file not found: {cred_file}')
    creds = {}
    with open(cred_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            creds[key] = val
    return creds


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_path: str) -> str:
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'


# ---------------------------------------------------------------------------
# Content validation
# ---------------------------------------------------------------------------

def is_valid_answer(content) -> bool:
    """Validate that response content is non-null and non-empty after strip."""
    if content is None:
        return False
    if not isinstance(content, str):
        return False
    if len(content.strip()) == 0:
        return False
    return True


def has_reasoning_only(completion) -> bool:
    """Detect reasoning-only response: reasoning_content present but no visible content."""
    msg = completion.choices[0].message
    content = msg.content
    reasoning = getattr(msg, 'reasoning_content', None)
    visible_empty = content is None or len(content.strip()) == 0
    has_reasoning = reasoning is not None and len(str(reasoning).strip()) > 0
    return visible_empty and has_reasoning


def build_follow_up_prompt(original_prompt: str) -> str:
    """Build a follow-up prompt requesting final visible answer after reasoning-only response."""
    return (
        f"{original_prompt}\n\n"
        f"[SYSTEM NOTE: Your previous response only contained reasoning with no final answer. "
        f"Please provide your FINAL ANSWER now. "
        f"Start directly with the answer, followed by brief reasoning if helpful.]"
    )


# ---------------------------------------------------------------------------
# Single-item generation with bounded retry
# ---------------------------------------------------------------------------

def generate_with_retry(
    client: OpenAI,
    model: str,
    prompt: str,
    gen_length: int,
    temperature: float,
    max_retries: int,
    qid: str,
) -> dict:
    """
    Generate an answer with bounded retry for null-content responses.

    Returns dict with {hypothesis, _usage} on success,
    or raises RuntimeError after retry exhaustion.
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency = 0.0
    current_prompt = prompt
    is_follow_up = False

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': current_prompt}],
            n=1,
            temperature=temperature,
            max_tokens=gen_length,
        )
        elapsed = time.time() - t0
        total_latency += elapsed

        usage = completion.usage
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens

        msg = completion.choices[0].message
        content = msg.content

        if is_valid_answer(content):
            answer = content.strip()
            cost_input = (total_prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT
            cost_output = (total_completion_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT

            print(f'  ✓ Attempt {attempt}: valid answer ({len(answer)} chars, '
                  f'{usage.completion_tokens} tokens, {elapsed:.1f}s)')

            return {
                'hypothesis': answer,
                '_usage': {
                    'prompt_tokens': total_prompt_tokens,
                    'completion_tokens': total_completion_tokens,
                    'total_tokens': total_prompt_tokens + total_completion_tokens,
                    'latency_s': round(total_latency, 2),
                    'attempts': attempt,
                    'had_follow_up': is_follow_up,
                },
                '_cost': {
                    'input_usd': round(cost_input, 8),
                    'output_usd': round(cost_output, 8),
                    'total_usd': round(cost_input + cost_output, 8),
                },
            }

        # Content is null/empty — diagnose why
        reason = 'null' if content is None else f'empty ({len(content)} chars after strip)'
        print(f'  ✗ Attempt {attempt}: {reason} content, {usage.completion_tokens} tokens, {elapsed:.1f}s')

        if has_reasoning_only(completion):
            reasoning_len = len(str(msg.reasoning_content))
            print(f'    Detected reasoning-only response ({reasoning_len} chars reasoning)')
            # Build follow-up prompt for next attempt
            current_prompt = build_follow_up_prompt(prompt)
            is_follow_up = True
        elif not is_follow_up:
            # First empty response — try follow-up anyway
            current_prompt = build_follow_up_prompt(prompt)
            is_follow_up = True

        # Continue to next retry

    # Exhausted all retries
    raise RuntimeError(
        f'Exhausted {max_retries} retries for {qid}: '
        f'{total_prompt_tokens}+{total_completion_tokens} tokens burned, no valid content'
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Credentials
    creds = load_credentials(args.cred_file)
    api_key = creds.get('OPENROUTER_API_KEY')
    api_base = creds.get('BENCHD_API_BASE', 'https://openrouter.ai/api/v1')
    if not api_key:
        print('ERROR: OPENROUTER_API_KEY not found', file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=api_base)

    # Load input
    with open(args.in_file) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f'Loaded {len(entries)} entries from {args.in_file}')

    # Hashes
    retrieval_hash = sha256_file(args.in_file)
    official_commit = git_commit(os.path.join(REPO_ROOT, 'LongMemEval'))
    model_max_length = args.max_context
    gen_length = args.gen_length
    max_retrieval_length = model_max_length - gen_length - 1000

    # Tokenizer
    tokenizer = tiktoken.get_encoding('o200k_base')

    # Output paths
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    out_prefix = os.path.basename(args.in_file).replace('.jsonl', '')
    hyp_file = os.path.join(
        args.out_dir,
        f'{out_prefix}_hypotheses_{args.model.replace("/", "_")}_repair_topk{args.topk}_{timestamp}.jsonl'
    )
    meta_file = hyp_file + '.repair-metadata.json'

    print(f'Model: {args.model}')
    print(f'Max retries: {args.max_retries}')
    print(f'Max context: {model_max_length}, Gen length: {gen_length}, TopK: {args.topk}')
    print(f'Output: {hyp_file}')
    print()

    # Generation loop
    hypotheses = []
    errors = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    for i, entry in enumerate(entries):
        qid = entry['question_id']
        question = entry['question'][:80]
        print(f'[{i+1}/{len(entries)}] {qid}: {question}...')

        try:
            # Prepare prompt
            prompt = prepare_prompt(
                entry=entry,
                retriever_type=args.retriever_type,
                topk_context=args.topk,
                useronly=args.useronly,
                history_format=args.history_format,
                cot=args.cot,
                tokenizer=tokenizer,
                tokenizer_backend='openai',
                max_retrieval_length=max_retrieval_length,
                merge_key_expansion_into_value='none',
            )

            prompt_tokens_est = len(tokenizer.encode(prompt, allowed_special={'<|endoftext|>'}))
            print(f'  Prompt tokens (est): {prompt_tokens_est}')

            # Generate with retry
            result = generate_with_retry(
                client=client,
                model=args.model,
                prompt=prompt,
                gen_length=gen_length,
                temperature=args.temperature,
                max_retries=args.max_retries,
                qid=qid,
            )

            hyp_entry = {
                'question_id': qid,
                'hypothesis': result['hypothesis'],
                '_usage': result['_usage'],
                '_cost': result['_cost'],
            }
            hypotheses.append(hyp_entry)

            total_prompt_tokens += result['_usage']['prompt_tokens']
            total_completion_tokens += result['_usage']['completion_tokens']
            total_cost += result['_cost']['total_usd']

            answer_preview = result['hypothesis'][:120]
            print(f'  Answer preview: {answer_preview}...')
            print()

        except Exception as e:
            error_msg = f'{type(e).__name__}: {e}'
            print(f'  ✗ FAILED: {error_msg}', file=sys.stderr)
            errors.append({
                'question_id': qid,
                'error': str(e),
            })
            print()
            continue

    # Write hypotheses JSONL
    with open(hyp_file, 'w') as f:
        for h in hypotheses:
            # Remove _cost from output to match original schema
            output = {k: v for k, v in h.items() if k != '_cost'}
            print(json.dumps(output), file=f)

    # Compute cost summary
    cost_summary = {
        'total_input_usd': round(
            (total_prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT, 8
        ),
        'total_output_usd': round(
            (total_completion_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT, 8
        ),
        'total_usd': round(total_cost, 8),
    }

    # Write metadata
    metadata = {
        'model': args.model,
        'openrouter_base_url': api_base,
        'max_context': model_max_length,
        'gen_length': gen_length,
        'retriever_type': args.retriever_type,
        'topk_context': args.topk,
        'history_format': args.history_format,
        'cot': args.cot,
        'temperature': args.temperature,
        'max_retries': args.max_retries,
        'official_commit': official_commit,
        'retrieval_file': args.in_file,
        'retrieval_hash_sha256': retrieval_hash,
        'total_entries': len(entries),
        'successful': len(hypotheses),
        'errors': len(errors),
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'total_tokens': total_prompt_tokens + total_completion_tokens,
        'cost_usd': cost_summary,
        'started_at': timestamp,
        'is_repair': True,
        'original_root_cause': 'content=None from DeepSeek R1 reasoning-only response, '
                               'line 202 .strip() on None',
        'repair_strategy': f'bounded retry ({args.max_retries} attempts) with follow-up prompt '
                           'requesting final visible answer when reasoning-only detected',
    }
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Report
    print('=' * 60)
    print(f'REPAIR COMPLETE: {len(hypotheses)}/{len(entries)} successful, {len(errors)} errors')
    print(f'Total prompt tokens:    {total_prompt_tokens}')
    print(f'Total completion tokens: {total_completion_tokens}')
    print(f'Total tokens:            {total_prompt_tokens + total_completion_tokens}')
    print(f'Total cost:              ${total_cost:.6f}')
    print(f'Hypotheses:  {hyp_file}')
    print(f'Metadata:    {meta_file}')

    if errors:
        print(f'\nErrors:')
        for e in errors:
            print(f'  {e["question_id"]}: {e["error"][:200]}')
        print(f'\nFAILED: {len(errors)} items could not be repaired after {args.max_retries} retries each.')
        sys.exit(1)

    # Verify all hypotheses non-empty
    for h in hypotheses:
        if not h['hypothesis'].strip():
            print(f'FATAL: empty hypothesis for {h["question_id"]} after repair', file=sys.stderr)
            sys.exit(1)

    print('All repair hypotheses non-empty. ✓')
    print(f'Cost breakdown: input=${cost_summary["total_input_usd"]:.6f}, '
          f'output=${cost_summary["total_output_usd"]:.6f}, '
          f'total=${cost_summary["total_usd"]:.6f}')


if __name__ == '__main__':
    main()
