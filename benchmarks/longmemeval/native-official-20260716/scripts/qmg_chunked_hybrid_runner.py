#!/usr/bin/env python3
"""
QMG Chunked Hybrid Retrieval Runner — LongMemEval Benchmark
===========================================================

Method: qmg-bm25-hybrid-70-30-chunked-session-v1.3

Implements the QMG published retrieval algorithm from the public benchmark:
  - 500-char chunks, 100-char overlap
  - gte-large embeddings (thenlper/gte-large)
  - BM25 on stopword-filtered tokens
  - 70/30 normalized min-max fusion
  - Per-session mean of top-3 chunk scores → final session ranking

Key guarantees:
  - ZERO REST API calls — pure local chunked hybrid retrieval
  - No gold labels/answer_session_ids influence ranking
  - All session IDs and dates preserved (no renaming)
  - has_answer stripped from any text handed to model
  - Deterministic: fixed seeds, no randomness
  - GPU-optimized: batch encoding, embedding cache
  - Output is official-compatible JSONL consumable by:
      prepare_prompt(..., retriever_type='flat-session')
      print_retrieval_metrics.py

Usage:
    python3 qmg_chunked_hybrid_runner.py \
        --in-file LongMemEval/data/longmemeval_s_cleaned.json \
        --out-file output/qmg_chunked_hybrid_retrieval.jsonl \
        --max-items 10

DK 🦍 — Jul 2026
"""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("qmg-chunked-hybrid")

# ── Constants ────────────────────────────────────────────────────────
METHOD_NAME = "qmg-bm25-hybrid-70-30-chunked-session-v1.3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
HYBRID_EMBED_WEIGHT = 0.7
HYBRID_BM25_WEIGHT = 0.3
TOP_CHUNKS_FOR_SESSION = 3
EMBED_BATCH_SIZE = 256
MODEL_NAME = "thenlper/gte-large"

# ── Determinism ──────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

# ── Stopwords (same as public benchmark) ─────────────────────────────
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'about', 'up', 'down', 'this',
    'that', 'these', 'those', 'it', 'its', 'and', 'but', 'or', 'if',
    'because', 'while', 'which', 'who', 'whom', 'what',
}


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="QMG Chunked Hybrid Retrieval Runner")
    p.add_argument("--in-file", required=True, help="Path to longmemeval_s_cleaned.json")
    p.add_argument("--out-file", required=True, help="Output JSONL path")
    p.add_argument("--max-items", type=int, default=500, help="Max items (default: 500)")
    p.add_argument("--skip-first", type=int, default=0, help="Skip first N items")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Device for embedding model")
    p.add_argument("--no-cache", action="store_true", help="Disable embedding cache")
    return p.parse_args()


# ── Text helpers ─────────────────────────────────────────────────────
def flatten_session_user_only(session_turns):
    """
    Flatten session turns to user-only text, exactly as the official
    LongMemEval code does for session-granularity flat indexing.

    Official process_item_flat_index:
        text = ' '.join([interact['content']
                         for interact in data if interact['role'] == 'user'])
    """
    return " ".join(
        turn["content"]
        for turn in session_turns
        if turn.get("role") == "user"
    )


def strip_has_answer(session_turns):
    """
    Return a deep copy of session turns with 'has_answer' removed from every turn.

    This is what official prepare_prompt does before handing text to the model.
    """
    cleaned = []
    for turn in session_turns:
        t = dict(turn)
        t.pop("has_answer", None)
        cleaned.append(t)
    return cleaned


def tokenize(text):
    """BM25 tokenization — lowercase, stopword filter, min length 2."""
    return [w for w in text.lower().split() if w not in STOPWORDS and len(w) > 1]


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Sliding-window chunking. Returns list of chunk strings."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks if chunks else [text]


# ── Embedding cache ─────────────────────────────────────────────────
class EmbeddingCache:
    """Simple hash-based cache for text→embedding mappings."""

    def __init__(self):
        self._cache = {}  # sha256(text) → np.ndarray

    def get(self, text):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self._cache.get(key)

    def put(self, text, embedding):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._cache[key] = embedding

    def size(self):
        return len(self._cache)


# ── Retrieval engine ─────────────────────────────────────────────────
class ChunkedHybridRetriever:
    """
    Implements the QMG chunked hybrid retrieval algorithm.

    1. Flatten sessions to user-only text
    2. Chunk at CHUNK_SIZE with CHUNK_OVERLAP
    3. Batch-encode chunks with gte-large
    4. Build BM25 index on chunks
    5. 70/30 normalized min-max fusion
    6. Per-session mean of top-3 chunk scores → session ranking

    ZERO gold-label influence on ranking. Gold labels used ONLY
    for computing retrieval metrics AFTER rankings are frozen.
    """

    def __init__(self, device="auto", use_cache=True):
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info(f"Loading {MODEL_NAME} on {device}...")
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        self.embed_dim = self.model.get_sentence_embedding_dimension()
        log.info(f"Model loaded: dim={self.embed_dim}")
        self.use_cache = use_cache
        self.cache = EmbeddingCache() if use_cache else None

    def encode_chunks(self, chunks, show_progress=False):
        """Batch-encode chunks with GPU. Returns (N, dim) float32 array."""
        if not chunks:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        if self.use_cache:
            # Check cache first
            uncached = []
            uncached_indices = []
            embeddings = [None] * len(chunks)
            for i, c in enumerate(chunks):
                cached = self.cache.get(c)
                if cached is not None:
                    embeddings[i] = cached
                else:
                    uncached.append(c)
                    uncached_indices.append(i)

            if uncached:
                new_embs = self.model.encode(
                    uncached,
                    normalize_embeddings=True,
                    batch_size=EMBED_BATCH_SIZE,
                    show_progress_bar=show_progress,
                )
                for idx, emb in zip(uncached_indices, new_embs):
                    embeddings[idx] = emb
                    self.cache.put(uncached[uncached_indices.index(idx)], emb)

            return np.array(embeddings, dtype=np.float32)
        else:
            return self.model.encode(
                chunks,
                normalize_embeddings=True,
                batch_size=EMBED_BATCH_SIZE,
                show_progress_bar=show_progress,
            )

    def retrieve(self, item):
        """
        Run chunked hybrid retrieval for one dataset item.

        Returns:
            retrieval_results dict with ranked_items and metrics.
            Gold labels used ONLY for metrics, never for ranking.
        """
        from rank_bm25 import BM25Okapi

        query = item["question"]
        sessions = item["haystack_sessions"]
        session_ids = item["haystack_session_ids"]
        session_dates = item["haystack_dates"]
        answer_ids = set(item.get("answer_session_ids", []))

        n_sessions = len(sessions)

        # ── Step 1: Flatten sessions (user-only) and chunk ──
        # Strip has_answer from all text — no gold leakage
        all_chunks = []
        chunk_to_session = []  # maps chunk index → session index

        for si in range(n_sessions):
            cleaned_turns = strip_has_answer(sessions[si])
            text = flatten_session_user_only(cleaned_turns)
            for chunk in chunk_text(text):
                all_chunks.append(chunk)
                chunk_to_session.append(si)

        if not all_chunks:
            # Fallback: empty retrieval
            return self._empty_result(query, session_ids, session_dates, answer_ids)

        # ── Step 2: Encode query and chunks ──
        q_emb = self.encode_chunks([query])[0]
        chunk_embs = self.encode_chunks(all_chunks)

        # ── Step 3: BM25 scores ──
        tokenized_chunks = [tokenize(c) for c in all_chunks]
        bm25_index = BM25Okapi(tokenized_chunks)
        bm25_raw = np.array(bm25_index.get_scores(tokenize(query)), dtype=np.float32)

        # ── Step 4: Embedding cosine scores ──
        chunk_scores = chunk_embs @ q_emb  # (n_chunks,)

        # ── Step 5: 70/30 normalized min-max fusion ──
        emb_min, emb_max = chunk_scores.min(), chunk_scores.max()
        e_range = emb_max - emb_min if emb_max != emb_min else 1.0
        bm25_min, bm25_max = bm25_raw.min(), bm25_raw.max()
        b_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
        fused = (HYBRID_EMBED_WEIGHT * ((chunk_scores - emb_min) / e_range) +
                 HYBRID_BM25_WEIGHT * ((bm25_raw - bm25_min) / b_range))

        # ── Step 6: Per-session score = mean of top-3 chunk scores ──
        session_chunk_scores = {}
        for ci, si in enumerate(chunk_to_session):
            if si not in session_chunk_scores:
                session_chunk_scores[si] = []
            session_chunk_scores[si].append(float(fused[ci]))

        session_scores = {}
        for si, scores in session_chunk_scores.items():
            top_scores = sorted(scores, reverse=True)[:TOP_CHUNKS_FOR_SESSION]
            session_scores[si] = float(np.mean(top_scores))

        # Rank sessions by descending score (deterministic tie-break by index)
        ranking = sorted(
            range(n_sessions),
            key=lambda i: (session_scores.get(i, -float("inf")), -i),
            reverse=True,
        )

        # ── Step 7: Build ranked_items ──
        ranked_items = []
        for si in ranking:
            # Use original user-only text for the ranked item
            cleaned_turns = strip_has_answer(sessions[si])
            text = flatten_session_user_only(cleaned_turns)
            ranked_items.append({
                "corpus_id": session_ids[si],
                "text": text,
                "timestamp": session_dates[si],
            })

        # ── Step 8: Metrics (gold labels used ONLY here) ──
        metrics = self._compute_metrics(ranking, answer_ids, session_ids)

        return {
            "query": query,
            "ranked_items": ranked_items,
            "metrics": metrics,
            "method": METHOD_NAME,
        }

    def _empty_result(self, query, session_ids, session_dates, answer_ids):
        """Return empty retrieval result."""
        return {
            "query": query,
            "ranked_items": [],
            "metrics": self._compute_metrics([], answer_ids, session_ids),
            "method": METHOD_NAME,
        }

    def _compute_metrics(self, ranking, answer_ids, session_ids):
        """Compute recall@K and NDCG@K metrics from frozen ranking."""
        correct_docs = list(answer_ids)
        metrics_session = {}
        for k in [1, 3, 5, 10, 30, 50]:
            recall_any, recall_all, ndcg_val = self._evaluate(ranking, correct_docs, session_ids, k)
            metrics_session[f"recall_any@{k}"] = float(recall_any)
            metrics_session[f"recall_all@{k}"] = float(recall_all)
            metrics_session[f"ndcg_any@{k}"] = float(ndcg_val)
        return {"session": metrics_session, "turn": {}}

    @staticmethod
    def _evaluate(rankings, correct_docs, corpus_ids, k):
        """Evaluate retrieval at k."""
        # Handle empty rankings
        effective_k = min(k, len(rankings))
        recalled_docs = set(corpus_ids[idx] for idx in rankings[:effective_k])
        recall_any = float(any(doc in recalled_docs for doc in correct_docs))
        recall_all = float(all(doc in recalled_docs for doc in correct_docs))

        # NDCG
        relevances = [1.0 if doc_id in correct_docs else 0.0 for doc_id in corpus_ids]
        sorted_rel = [relevances[idx] for idx in rankings[:effective_k]]
        ideal_rel = sorted(relevances, reverse=True)

        def dcg(rels, kk):
            rels = rels[:kk]
            if not rels:
                return 0.0
            score = rels[0]
            for i in range(1, len(rels)):
                score += rels[i] / math.log2(i + 2)
            return score

        ideal_dcg = dcg(ideal_rel, k)
        actual_dcg = dcg(sorted_rel, k)
        ndcg_score = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        return recall_any, recall_all, ndcg_score


# ── Dataset helpers ──────────────────────────────────────────────────
def load_dataset(path):
    with open(path) as f:
        data = json.load(f)
    log.info(f"Loaded {len(data)} items from {path}")
    return data


def hash_dataset(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_code():
    """Hash this runner's source for reproducibility."""
    h = hashlib.sha256()
    with open(__file__, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


# ── Main ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Load dataset
    dataset = load_dataset(args.in_file)
    dataset_hash = hash_dataset(args.in_file)
    code_hash = hash_code()

    # Save config
    config = {
        "method": METHOD_NAME,
        "model": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "hybrid_ratio": f"{int(HYBRID_EMBED_WEIGHT*100)}% embedding + {int(HYBRID_BM25_WEIGHT*100)}% BM25",
        "top_chunks_per_session": TOP_CHUNKS_FOR_SESSION,
        "embed_batch_size": EMBED_BATCH_SIZE,
        "dataset": os.path.abspath(args.in_file),
        "dataset_hash": dataset_hash,
        "code_hash": code_hash,
        "random_seed": RANDOM_SEED,
        "device": args.device,
        "max_items": min(args.max_items, len(dataset)),
        "skip_first": args.skip_first,
        "started_at": datetime.now().isoformat(),
        "runner_version": "1.3.0",
    }
    config_path = args.out_file + ".config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Config saved to {config_path}")
    log.info(f"Dataset hash: {dataset_hash}")
    log.info(f"Code hash: {code_hash}")

    # Initialize retriever
    retriever = ChunkedHybridRetriever(
        device=args.device,
        use_cache=not args.no_cache,
    )

    # Process items
    items = dataset[args.skip_first : args.skip_first + args.max_items]
    log.info(f"Processing {len(items)} items")

    start_time = time.time()
    processed = 0
    error_count = 0

    with open(args.out_file, "w") as outf:
        for i, item in enumerate(items):
            qid = item["question_id"]
            try:
                rr = retriever.retrieve(item)
            except Exception as e:
                log.error(f"[{qid}] FAILED: {e}", exc_info=True)
                error_count += 1
                rr = {
                    "query": item["question"],
                    "ranked_items": [],
                    "metrics": {"session": {}, "turn": {}},
                    "method": METHOD_NAME,
                    "error": str(e),
                }

            # Build full output entry
            output_entry = {
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "question": item["question"],
                "answer": item["answer"],
                "question_date": item["question_date"],
                "haystack_dates": item["haystack_dates"],
                "haystack_sessions": item["haystack_sessions"],
                "haystack_session_ids": item["haystack_session_ids"],
                "answer_session_ids": item["answer_session_ids"],
                "retrieval_results": rr,
            }

            outf.write(json.dumps(output_entry) + "\n")
            outf.flush()

            processed += 1
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0

            if processed % 10 == 0 or processed == len(items):
                if rate > 0:
                    eta = (len(items) - processed) / rate
                    log.info(
                        f"Progress: {processed}/{len(items)} "
                        f"({rate:.1f} items/s, ETA {eta/60:.1f}m)"
                    )
                else:
                    log.info(f"Progress: {processed}/{len(items)}")

    total_elapsed = time.time() - start_time
    log.info(f"Done. {processed} items in {total_elapsed:.1f}s "
             f"({total_elapsed/processed:.1f}s/item)")
    if error_count:
        log.warning(f"Errors: {error_count}")
    log.info(f"Output: {args.out_file}")
    log.info(f"Config: {config_path}")

    # Quick aggregate metrics
    print_aggregate_metrics(args.out_file)


def print_aggregate_metrics(out_file):
    """Print aggregate metrics summary from output file."""
    all_metrics = []
    with open(out_file) as f:
        for line in f:
            entry = json.loads(line)
            if "_abs" not in entry["question_id"]:
                rr = entry.get("retrieval_results", {})
                metrics = rr.get("metrics", {}).get("session", {})
                if metrics:
                    all_metrics.append(metrics)

    if not all_metrics:
        log.warning("No metrics to aggregate")
        return

    log.info("=" * 70)
    log.info(f"Aggregate Retrieval Metrics — {len(all_metrics)} items")
    log.info(f"Method: {METHOD_NAME}")
    log.info("-" * 70)
    for k in [1, 3, 5, 10]:
        line_parts = []
        for m in ["recall_any", "recall_all", "ndcg_any"]:
            vals = [x.get(f"{m}@{k}", 0) for x in all_metrics]
            if vals:
                avg = np.mean(vals) * 100
                line_parts.append(f"{m}@{k}={avg:.1f}%")
        log.info("  " + "  ".join(line_parts))
    log.info("=" * 70)


if __name__ == "__main__":
    main()