#!/usr/bin/env python3
"""
LongMemEval Benchmark — BM25 Hybrid Retrieval (Jul 2026)

Compares embedding-only (gte-large chunked) vs BM25 hybrid (70/30 fusion)
on the official LongMemEval dataset (xiaowu0162/longmemeval, longmemeval_s split).

Usage:
    pip install quantum-memory-graph rank-bm25 sentence-transformers
    python benchmarks/run_longmemeval_hybrid.py --limit 50   # Quick test
    python benchmarks/run_longmemeval_hybrid.py --force       # Full 500

Requires the LongMemEval dataset. Download from HuggingFace:
    from datasets import load_dataset
    ds = load_dataset('xiaowu0162/longmemeval', split='longmemeval_s')
    # Save to longmemeval_s.json

Or use the preprocessed version (sessions flattened to text).

DK 🦍
"""
import json, time, math, sys, os, argparse
import numpy as np

# English stopwords for BM25 tokenization
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

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def tokenize(text):
    return [w for w in text.lower().split() if w not in STOPWORDS and len(w) > 1]


def flatten_session(session):
    if isinstance(session, str):
        return session
    if isinstance(session, list):
        parts = []
        for turn in session:
            if isinstance(turn, dict):
                parts.append(f"{turn.get('role', '')}: {turn.get('content', turn.get('text', str(turn)))}")
            else:
                parts.append(str(turn))
        return "\n".join(parts)
    return str(session)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks if chunks else [text]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to test")
    parser.add_argument("--force", action="store_true", help="Run full 500 questions")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset JSON")
    args = parser.parse_args()

    # Try common paths
    data_paths = [
        args.data,
        "longmemeval_s.json",
        os.path.expanduser("~/projects-shared/LongMemEval/data/longmemeval_s_cleaned.json"),
    ]
    data_path = None
    for p in data_paths:
        if p and os.path.exists(p):
            data_path = p
            break
    
    if not data_path:
        print("ERROR: LongMemEval dataset not found.")
        print("Download from: https://huggingface.co/datasets/xiaowu0162/longmemeval")
        print("Or pass --data /path/to/longmemeval_s.json")
        sys.exit(1)

    print(f"Loading {data_path}...", flush=True)
    with open(data_path) as f:
        data = json.load(f)
    
    limit = args.limit
    if args.force:
        limit = None
    if limit:
        data = data[:limit]
    
    print(f"Running on {len(data)} questions", flush=True)

    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi

    model = SentenceTransformer("thenlper/gte-large", device="cuda" if __import__('torch').cuda.is_available() else "cpu")
    print(f"Model: thenlper/gte-large, dim={model.get_sentence_embedding_dimension()}", flush=True)

    results = {"embedding": {}, "hybrid": {}}
    per_question = []
    n_valid = 0
    t_start = time.time()

    for idx, item in enumerate(data):
        question = item.get("question", item.get("query", ""))
        haystack = item.get("haystack_sessions", item.get("sessions", item.get("corpus", [])))
        haystack_ids = item.get("haystack_session_ids", item.get("session_ids", []))
        answer_ids = item.get("answer_session_ids", item.get("answer_ids", []))

        gold_indices = []
        for g in answer_ids:
            try:
                gold_indices.append(haystack_ids.index(g))
            except ValueError:
                pass

        if not gold_indices or len(haystack) < 3:
            continue

        n_sessions = len(haystack)

        # Chunk sessions
        all_chunks, chunk_to_session = [], []
        for si, sess in enumerate(haystack):
            text = flatten_session(sess)
            for c in chunk_text(text):
                all_chunks.append(c)
                chunk_to_session.append(si)

        # Encode
        q_emb = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        chunk_embs = model.encode(all_chunks, normalize_embeddings=True, batch_size=256, show_progress_bar=False)

        # BM25
        tokenized_chunks = [tokenize(c) for c in all_chunks]
        bm25_index = BM25Okapi(tokenized_chunks)
        bm25_raw = bm25_index.get_scores(tokenize(question))

        # --- Embedding-only scoring ---
        chunk_scores = chunk_embs @ q_emb
        
        def session_score(chunk_scores_array, chunk_to_session_map):
            sess_scores = {}
            for ci, si in enumerate(chunk_to_session_map):
                if si not in sess_scores:
                    sess_scores[si] = []
                sess_scores[si].append(float(chunk_scores_array[ci]))
            for si in sess_scores:
                sc = sorted(sess_scores[si], reverse=True)
                sess_scores[si] = np.mean(sc[:min(3, len(sc))])
            return sess_scores

        emb_session = session_score(chunk_scores, chunk_to_session)
        emb_ranking = sorted(range(n_sessions), key=lambda i: emb_session.get(i, -1), reverse=True)

        # --- BM25 hybrid scoring (70/30) ---
        emb_min, emb_max = chunk_scores.min(), chunk_scores.max()
        e_range = emb_max - emb_min if emb_max != emb_min else 1.0
        bm25_min, bm25_max = bm25_raw.min(), bm25_raw.max()
        b_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
        fused = 0.7 * ((chunk_scores - emb_min) / e_range) + 0.3 * ((bm25_raw - bm25_min) / b_range)

        hyb_session = session_score(fused, chunk_to_session)
        hyb_ranking = sorted(range(n_sessions), key=lambda i: hyb_session.get(i, -1), reverse=True)

        # --- Metrics ---
        gold_set = set(gold_indices)

        def compute_metrics(ranking, gold_set, method_name):
            m = {"method": method_name, "question_idx": idx}
            for K in [1, 5, 10]:
                m[f"R@{K}"] = 1.0 if set(ranking[:K]) & gold_set else 0.0
            dcg = sum(1.0 / math.log2(i + 2) for i, r_idx in enumerate(ranking[:5]) if r_idx in gold_set)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), 5)))
            m["NDCG@5"] = dcg / idcg if idcg > 0 else 0.0
            return m

        per_question.append(compute_metrics(emb_ranking, gold_set, "embedding-only"))
        per_question.append(compute_metrics(hyb_ranking, gold_set, "bm25-hybrid-70-30"))
        n_valid += 1

        if n_valid % 100 == 0:
            elapsed = time.time() - t_start
            emb_r5 = sum(1 for r in per_question if r["method"] == "embedding-only" and r["R@5"] > 0) / n_valid * 100
            hyb_r5 = sum(1 for r in per_question if r["method"] == "bm25-hybrid-70-30" and r["R@5"] > 0) / n_valid * 100
            print(f"\r  [{n_valid}/{len(data)} valid, {elapsed:.0f}s] emb R@5={emb_r5:.1f}% hyb R@5={hyb_r5:.1f}%", end="", flush=True)

    elapsed = time.time() - t_start
    print()
    print()

    # Aggregate
    agg = {}
    for method in ["embedding-only", "bm25-hybrid-70-30"]:
        method_results = [r for r in per_question if r["method"] == method]
        agg[method] = {}
        for metric in ["R@1", "R@5", "R@10", "NDCG@5"]:
            agg[method][metric] = round(sum(r[metric] for r in method_results) / len(method_results) * 100, 2)

    print("=" * 60)
    print(f"LongMemEval — {n_valid} questions — {elapsed:.0f}s")
    print("=" * 60)
    print(f"{'':20s} {'gte-large only':>15s} {'+BM25 Hybrid':>15s} {'Δ':>8s}")
    print("-" * 60)
    for metric in ["R@1", "R@5", "R@10", "NDCG@5"]:
        emb = agg["embedding-only"][metric]
        hyb = agg["bm25-hybrid-70-30"][metric]
        delta = hyb - emb
        print(f"{metric:20s} {emb:14.1f}% {hyb:14.1f}% {delta:+7.1f}%")
    print("=" * 60)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__) or ".", "longmemeval_hybrid_results.json")
    output = {
        "benchmark": "LongMemEval",
        "dataset": data_path,
        "model": "thenlper/gte-large",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "hybrid_ratio": "70% embedding + 30% BM25",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": int(elapsed),
        "results": agg,
        "per_question": per_question,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
