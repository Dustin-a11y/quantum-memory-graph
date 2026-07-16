#!/usr/bin/env python3
"""
LongMemEval Generation Wrapper — OpenRouter DeepSeek-R1

External wrapper that calls the official prepare_prompt() from
LongMemEval/src/generation/run_generation.py and sends prompts to
OpenRouter's deepseek/deepseek-r1 via OpenAI-compatible client.

This avoids modifying the official repo while supporting model IDs
not present in its hardcoded model2maxlength map.

Usage:
    python run_generation_openrouter.py \
        --in-file runs/native-official-20260715/qmg_retrieval_10.jsonl \
        --out-dir runs/native-official-20260715 \
        --cred-file ./benchd-openrouter.env

Credentials are sourced from --cred-file at runtime and NEVER stored in artifacts.
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
# Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='LongMemEval generation via OpenRouter DeepSeek-R1')
    parser.add_argument('--in-file', required=True, help='JSONL input file with retrieval results')
    parser.add_argument('--out-dir', required=True, help='Output directory for hypotheses + metadata')
    parser.add_argument('--cred-file', required=True, help='Path to .env file with OPENROUTER_API_KEY, BENCHD_API_BASE, etc.')
    parser.add_argument('--model', default='deepseek/deepseek-r1', help='OpenRouter model ID')
    parser.add_argument('--max-context', type=int, default=64000, help='Conservative max context length (tokens)')
    parser.add_argument('--topk', type=int, default=10, help='Top-K retrieved chunks')
    parser.add_argument('--gen-length', type=int, default=800, help='Max generation tokens')
    parser.add_argument('--retriever-type', default='flat-session')
    parser.add_argument('--history-format', default='nl')
    parser.add_argument('--useronly', action='store_true', default=False)
    parser.add_argument('--cot', action='store_true', default=True)
    parser.add_argument('--temperature', type=float, default=0.0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Credential loading — read at runtime, never store
# ---------------------------------------------------------------------------

def load_credentials(cred_file: str) -> dict:
    """Load credentials from a .env-style file. Returns dict without logging values."""
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
# Hashing helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_path: str) -> str:
    """Get current HEAD commit hash from a git repo."""
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
# Tokenizer setup — use tiktoken as proxy for OpenRouter models
# ---------------------------------------------------------------------------

def get_tokenizer():
    """Return (tokenizer, backend) tuple compatible with prepare_prompt."""
    return tiktoken.get_encoding('o200k_base'), 'openai'


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load credentials
    creds = load_credentials(args.cred_file)
    api_key = creds.get('OPENROUTER_API_KEY')
    api_base = creds.get('BENCHD_API_BASE', 'https://openrouter.ai/api/v1')
    if not api_key:
        print('ERROR: OPENROUTER_API_KEY not found in credential file', file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=api_base)

    # Load input data
    with open(args.in_file) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f'Loaded {len(entries)} entries from {args.in_file}')

    # Compute hashes for provenance
    retrieval_hash = sha256_file(args.in_file)
    official_commit = git_commit(os.path.join(REPO_ROOT, 'LongMemEval'))
    model_max_length = args.max_context
    gen_length = args.gen_length
    max_retrieval_length = model_max_length - gen_length - 1000

    # Tokenizer
    tokenizer, tokenizer_backend = get_tokenizer()

    # Output paths
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    out_prefix = os.path.basename(args.in_file).replace('.jsonl', '')
    hyp_file = os.path.join(args.out_dir,
                            f'{out_prefix}_hypotheses_{args.model.replace("/", "_")}_topk{args.topk}_{timestamp}.jsonl')
    meta_file = hyp_file + '.metadata.json'

    # Generation
    total_prompt_tokens = 0
    total_completion_tokens = 0
    hypotheses = []
    errors = []

    print(f'Model: {args.model}')
    print(f'Max context: {model_max_length}, Gen length: {gen_length}, TopK: {args.topk}')
    print(f'Retriever: {args.retriever_type}, Format: {args.history_format}, '
          f'UserOnly: {args.useronly}, CoT: {args.cot}')
    print(f'Output: {hyp_file}')
    print()

    for i, entry in enumerate(entries):
        qid = entry['question_id']
        print(f'[{i+1}/{len(entries)}] Processing {qid}: {entry["question"][:80]}...')

        try:
            # Prepare prompt using official function
            prompt = prepare_prompt(
                entry=entry,
                retriever_type=args.retriever_type,
                topk_context=args.topk,
                useronly=args.useronly,
                history_format=args.history_format,
                cot=args.cot,
                tokenizer=tokenizer,
                tokenizer_backend=tokenizer_backend,
                max_retrieval_length=max_retrieval_length,
                merge_key_expansion_into_value='none',
            )

            prompt_tokens_est = len(tokenizer.encode(prompt, allowed_special={'<|endoftext|>'}))
            print(f'  Prompt tokens (est): {prompt_tokens_est}')

            # API call
            t0 = time.time()
            completion = client.chat.completions.create(
                model=args.model,
                messages=[{'role': 'user', 'content': prompt}],
                n=1,
                temperature=args.temperature,
                max_tokens=gen_length,
            )
            elapsed = time.time() - t0

            answer = completion.choices[0].message.content
            # Guard against DeepSeek R1 reasoning-only responses where content=None
            if answer is None or not answer.strip():
                print(f'  WARNING: null/empty content for {qid}, skipping (use repair wrapper)', file=sys.stderr)
                error_msg = f'Null/empty content for {qid}: content={repr(answer)[:100]}'
                print(f'  {error_msg}', file=sys.stderr)
                errors.append({'question_id': qid, 'error': error_msg})
                continue
            answer = answer.strip()
            usage = completion.usage

            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens

            cost = None
            if hasattr(usage, 'prompt_tokens_details'):
                cost = getattr(usage.prompt_tokens_details, 'cost', None)

            hyp_entry = {
                'question_id': qid,
                'hypothesis': answer,
                '_usage': {
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens,
                    'latency_s': round(elapsed, 2),
                }
            }
            hypotheses.append(hyp_entry)

            print(f'  Hypothesis ({usage.completion_tokens} tokens, {elapsed:.1f}s): {answer[:120]}...')
            print()

        except Exception as e:
            error_msg = f'ERROR for {qid}: {type(e).__name__}: {e}'
            print(f'  {error_msg}', file=sys.stderr)
            errors.append({'question_id': qid, 'error': str(e)})
            # fail-closed: skip this entry
            continue

    # Write hypotheses
    with open(hyp_file, 'w') as f:
        for h in hypotheses:
            print(json.dumps(h), file=f)

    # Write metadata (NO secrets)
    metadata = {
        'model': args.model,
        'model_alias': args.model,
        'openrouter_base_url': api_base,
        'max_context': model_max_length,
        'gen_length': gen_length,
        'retriever_type': args.retriever_type,
        'topk_context': args.topk,
        'history_format': args.history_format,
        'useronly': args.useronly,
        'cot': args.cot,
        'temperature': args.temperature,
        'tokenizer_backend': 'tiktoken-o200k_base-proxy',
        'official_commit': official_commit,
        'retrieval_file': args.in_file,
        'retrieval_hash_sha256': retrieval_hash,
        'total_entries': len(entries),
        'successful': len(hypotheses),
        'errors': len(errors),
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'total_tokens': total_prompt_tokens + total_completion_tokens,
        'started_at': timestamp,
    }
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Report
    print('=' * 60)
    print(f'DONE: {len(hypotheses)}/{len(entries)} successful, {len(errors)} errors')
    print(f'Total prompt tokens:   {total_prompt_tokens}')
    print(f'Total completion tokens: {total_completion_tokens}')
    print(f'Total tokens:           {total_prompt_tokens + total_completion_tokens}')
    print(f'Hypotheses:  {hyp_file}')
    print(f'Metadata:    {meta_file}')

    if errors:
        print(f'\nErrors:')
        for e in errors:
            print(f'  {e["question_id"]}: {e["error"][:200]}')
        sys.exit(1)

    # Verify all hypotheses non-empty
    for h in hypotheses:
        if not h['hypothesis'].strip():
            print(f'WARNING: empty hypothesis for {h["question_id"]}', file=sys.stderr)
            sys.exit(1)

    print('All hypotheses non-empty. ✓')
    print('Metadata written (no secrets). ✓')


if __name__ == '__main__':
    main()
