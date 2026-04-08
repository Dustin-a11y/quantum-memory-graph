"""
LongMemEval v7 — Test more embedding models to beat MemPalace 96.6%.
Models: gte-large-en-v1.5, jina-embeddings-v3 (if available)

DK 🦍
"""

import json, sys, os, time, math, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer


def flatten_session(session):
    if isinstance(session, str):
        return session
    if isinstance(session, list):
        parts = []
        for turn in session:
            if isinstance(turn, dict):
                role = turn.get("role", "")
                content = turn.get("content", turn.get("text", str(turn)))
                parts.append(f"{role}: {content}" if role else content)
            else:
                parts.append(str(turn))
        return "\n".join(parts)
    return str(session)


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks if chunks else [text]


def recall_at_k(retrieved, gold, K):
    return len(set(retrieved[:K]) & set(gold)) / len(gold) if gold else 1.0

def ndcg_at_k(retrieved, gold, K):
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    dcg = sum(1.0 / math.log2(i + 2) for i, idx in enumerate(retrieved[:K]) if idx in gold_set)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), K)))
    return dcg / idcg if idcg > 0 else 0.0


def run_model(model_name, data, query_prefix=""):
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model_name}", flush=True)
    if query_prefix:
        print(f"Query prefix: '{query_prefix[:50]}...'", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"  Loading model...", flush=True)
    model = SentenceTransformer(model_name, trust_remote_code=True)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Dimension: {dim}", flush=True)

    t0 = time.time()
    question_data = []
    all_chunks = []

    for qi, item in enumerate(data):
        q_chunks = []
        q_map = []
        for si, sess in enumerate(item["haystack_sessions"]):
            text = flatten_session(sess)
            for c in chunk_text(text):
                q_chunks.append(c)
                q_map.append(si)
                all_chunks.append(c)
        question_data.append({
            "chunks": q_chunks, "map": q_map,
            "gold_ids": item["answer_session_ids"],
            "all_ids": item["haystack_session_ids"],
            "question": item["question"],
        })

    # Sanitize: truncate chunks to model max and strip non-printable
    max_len = 512 if 'gte' in model_name.lower() or 'e5' in model_name.lower() else 500
    all_chunks = [c[:max_len*4] for c in all_chunks]  # rough char limit

    print(f"  Total chunks: {len(all_chunks)}", flush=True)
    print(f"  Embedding chunks...", flush=True)
    try:
        all_embs = model.encode(all_chunks, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    except RuntimeError as e:
        print(f"  GPU failed ({e}), falling back to CPU...", flush=True)
        model = SentenceTransformer(model_name, device='cpu', trust_remote_code=True)
        all_embs = model.encode(all_chunks, normalize_embeddings=True, batch_size=256, show_progress_bar=True)

    q_texts = [item["question"] for item in data]
    if query_prefix:
        q_texts = [query_prefix + q for q in q_texts]
    print(f"  Embedding questions...", flush=True)
    q_embs = model.encode(q_texts, normalize_embeddings=True, batch_size=256)

    idx = 0
    for qd in question_data:
        n = len(qd["chunks"])
        qd["embs"] = all_embs[idx:idx+n]
        idx += n

    embed_time = time.time() - t0
    print(f"  Embed time: {embed_time:.0f}s", flush=True)

    t0 = time.time()
    r5 = r10 = n10 = 0
    for qi, qd in enumerate(question_data):
        gold = [qd["all_ids"].index(g) for g in qd["gold_ids"] if g in qd["all_ids"]]
        scores = np.dot(qd["embs"], q_embs[qi])
        session_scores = {}
        for ci, score in enumerate(scores):
            si = qd["map"][ci]
            if si not in session_scores:
                session_scores[si] = []
            session_scores[si].append(float(score))
        session_robust = {si: np.mean(sorted(cs, reverse=True)[:3]) for si, cs in session_scores.items()}
        ranked = sorted(session_robust.keys(), key=lambda s: session_robust[s], reverse=True)

        r5 += recall_at_k(ranked, gold, 5)
        r10 += recall_at_k(ranked, gold, 10)
        n10 += ndcg_at_k(ranked, gold, 10)

        if (qi+1) % 100 == 0:
            n = qi+1
            print(f"  [{n}/{len(data)}] R@5={r5/n:.3f} R@10={r10/n:.3f} NDCG={n10/n:.3f}", flush=True)

    n = len(data)
    result = {"model": model_name, "dim": dim,
              "recall_at_5": r5/n, "recall_at_10": r10/n, "ndcg_at_10": n10/n}
    print(f"\n  FINAL: R@5={result['recall_at_5']:.3f}  R@10={result['recall_at_10']:.3f}  NDCG={result['ndcg_at_10']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    del model, all_embs, all_chunks
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]
    print(f"Running {len(data)} questions\n", flush=True)

    models = [
        ("intfloat/e5-large-v2", "query: "),
        ("nomic-ai/nomic-embed-text-v1.5", "search_query: "),
        ("thenlper/gte-large", ""),
    ]

    results = []
    for model_name, prefix in models:
        try:
            r = run_model(model_name, data, query_prefix=prefix)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"ALL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\n  {'Model':<45} {'Dim':>5} {'R@5':>8} {'R@10':>8} {'NDCG':>8}", flush=True)
    print(f"  {'-'*45} {'-'*5} {'-'*8} {'-'*8} {'-'*8}", flush=True)

    # Include previous results
    print(f"  {'all-MiniLM-L6-v2':<45} {'384':>5} {'0.934':>8} {'0.974':>8} {'0.908':>8}", flush=True)
    print(f"  {'bge-large-en-v1.5':<45} {'1024':>5} {'0.959':>8} {'0.982':>8} {'0.940':>8}", flush=True)
    for r in results:
        name = r['model'].split('/')[-1]
        print(f"  {name:<45} {r['dim']:>5} {r['recall_at_5']:>7.3f} {r['recall_at_10']:>7.3f} {r['ndcg_at_10']:>7.3f}", flush=True)
    print(f"  {'MemPalace raw':<45} {'N/A':>5} {'0.966':>8} {'0.982':>8} {'0.889':>8}", flush=True)

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_longmemeval_v7.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {outpath}", flush=True)


if __name__ == "__main__":
    main()
