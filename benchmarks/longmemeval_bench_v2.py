"""
LongMemEval Benchmark v2 — Chunked sessions for denser graph.
Each conversation session is split into ~500 char chunks.
Graph+QAOA finds the best connected subgraph of chunks,
then maps back to session IDs for scoring.

DK 🦍
"""

import json, sys, os, time, math, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum_memory_graph.graph import MemoryGraph

print("Loading embedding model (cached)...", flush=True)
from sentence_transformers import SentenceTransformer
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready.", flush=True)


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
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 50:  # Skip tiny trailing chunks
            chunks.append(chunk)
        start = end - overlap
    return chunks if chunks else [text]


def recall_at_k(retrieved_indices, gold_indices, K):
    retrieved_set = set(retrieved_indices[:K])
    gold_set = set(gold_indices)
    if not gold_set:
        return 1.0
    return len(retrieved_set & gold_set) / len(gold_set)


def ndcg_at_k(retrieved_indices, gold_indices, K):
    gold_set = set(gold_indices)
    if not gold_set:
        return 1.0
    dcg = sum(1.0 / math.log2(i + 2) for i, idx in enumerate(retrieved_indices[:K]) if idx in gold_set)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), K)))
    return dcg / idcg if idcg > 0 else 0.0


def topk_chunked(sessions_text, session_indices, query, K):
    """Top-K on chunks, map back to sessions, return top K unique sessions."""
    all_chunks = []
    chunk_to_session = []
    for si, text in enumerate(sessions_text):
        chunks = chunk_text(text)
        for c in chunks:
            all_chunks.append(c)
            chunk_to_session.append(si)
    
    texts = all_chunks + [query]
    embs = MODEL.encode(texts, normalize_embeddings=True, batch_size=64)
    scores = np.dot(embs[:-1], embs[-1])
    
    # Aggregate: best chunk score per session
    session_scores = {}
    for ci, score in enumerate(scores):
        si = chunk_to_session[ci]
        if si not in session_scores or score > session_scores[si]:
            session_scores[si] = score
    
    ranked = sorted(session_scores.keys(), key=lambda s: session_scores[s], reverse=True)
    return ranked[:K*2]  # Return extra for R@10


def graph_qaoa_chunked(sessions_text, session_indices, query, K,
                        alpha=0.3, beta=0.25, gamma=0.15, threshold=0.15):
    """Build graph on chunks, QAOA selects best subgraph, map back to sessions."""
    all_chunks = []
    chunk_to_session = []
    for si, text in enumerate(sessions_text):
        chunks = chunk_text(text)
        for c in chunks:
            all_chunks.append(c)
            chunk_to_session.append(si)
    
    # Limit chunks to avoid explosion (take top 200 by similarity to query)
    if len(all_chunks) > 200:
        q_emb = MODEL.encode([query], normalize_embeddings=True)[0]
        c_embs = MODEL.encode(all_chunks, normalize_embeddings=True, batch_size=64)
        sims = np.dot(c_embs, q_emb)
        top_indices = np.argsort(sims)[-200:]
        filtered_chunks = [all_chunks[i] for i in top_indices]
        filtered_map = [chunk_to_session[i] for i in top_indices]
        all_chunks = filtered_chunks
        chunk_to_session = filtered_map
    
    g = MemoryGraph(similarity_threshold=threshold)
    g._embedder = MODEL
    g.add_memories_batch(all_chunks)
    
    from quantum_memory_graph.pipeline import recall as qrecall, set_graph
    set_graph(g)
    result = qrecall(query, K=K*3, graph=g, alpha=alpha, beta_conn=beta, gamma_cov=gamma)
    
    # Map selected chunks back to sessions
    session_hit_count = {}
    session_first_rank = {}
    for rank, sel in enumerate(result.get('memories', [])):
        for ci, c in enumerate(all_chunks):
            if c == sel['text']:
                si = chunk_to_session[ci]
                session_hit_count[si] = session_hit_count.get(si, 0) + 1
                if si not in session_first_rank:
                    session_first_rank[si] = rank
                break
    
    # Rank sessions by: hit count (desc), then first appearance (asc)
    ranked = sorted(session_hit_count.keys(),
                    key=lambda s: (-session_hit_count[s], session_first_rank.get(s, 999)))
    return ranked[:K*2]


def run_benchmark(data, method="topk_chunked", K=5, limit=None, **kwargs):
    if limit:
        data = data[:limit]
    
    r5 = 0; r10 = 0; n10 = 0; n = len(data)
    
    for i, item in enumerate(data):
        question = item["question"]
        sessions = item["haystack_sessions"]
        gold_ids = item["answer_session_ids"]
        all_ids = item["haystack_session_ids"]
        
        sessions_text = [flatten_session(s) for s in sessions]
        gold_indices = [all_ids.index(gid) for gid in gold_ids if gid in all_ids]
        
        if method == "topk_chunked":
            retrieved = topk_chunked(sessions_text, list(range(len(sessions))), question, K)
        else:
            retrieved = graph_qaoa_chunked(sessions_text, list(range(len(sessions))), question, K, **kwargs)
        
        r5 += recall_at_k(retrieved, gold_indices, 5)
        r10 += recall_at_k(retrieved, gold_indices, 10)
        n10 += ndcg_at_k(retrieved, gold_indices, 10)
        
        if (i+1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{n}] R@5={r5/(i+1):.3f} R@10={r10/(i+1):.3f} NDCG@10={n10/(i+1):.3f}", flush=True)
    
    return {"recall_at_5": r5/n, "recall_at_10": r10/n, "ndcg_at_10": n10/n, "n": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method", choices=["topk", "graph", "both"], default="both")
    args = parser.parse_args()
    
    print("Loading LongMemEval data...", flush=True)
    with open(args.data) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions\n", flush=True)
    
    results = {}
    
    if args.method in ("topk", "both"):
        print("=" * 60, flush=True)
        print("Top-K Chunked (500 char chunks, best-chunk-per-session)", flush=True)
        print("=" * 60, flush=True)
        t0 = time.time()
        results["topk_chunked"] = run_benchmark(data, "topk_chunked", limit=args.limit)
        r = results["topk_chunked"]
        print(f"\n  R@5:  {r['recall_at_5']:.3f}  R@10: {r['recall_at_10']:.3f}  NDCG@10: {r['ndcg_at_10']:.3f}  Time: {time.time()-t0:.0f}s\n", flush=True)
    
    if args.method in ("graph", "both"):
        print("=" * 60, flush=True)
        print("Graph+QAOA Chunked (top-200 chunks → graph → QAOA)", flush=True)
        print("=" * 60, flush=True)
        t0 = time.time()
        results["graph_chunked"] = run_benchmark(data, "graph_chunked", limit=args.limit,
                                                  alpha=0.3, beta=0.25, gamma=0.15, threshold=0.15)
        r = results["graph_chunked"]
        print(f"\n  R@5:  {r['recall_at_5']:.3f}  R@10: {r['recall_at_10']:.3f}  NDCG@10: {r['ndcg_at_10']:.3f}  Time: {time.time()-t0:.0f}s\n", flush=True)
    
    # Comparison
    if "topk_chunked" in results and "graph_chunked" in results:
        t = results["topk_chunked"]; g = results["graph_chunked"]
        print("=" * 60, flush=True)
        print("COMPARISON (chunked)", flush=True)
        print("=" * 60, flush=True)
        print(f"  Top-K Chunked:    R@5={t['recall_at_5']:.3f}  R@10={t['recall_at_10']:.3f}", flush=True)
        print(f"  Graph+QAOA:       R@5={g['recall_at_5']:.3f}  R@10={g['recall_at_10']:.3f}", flush=True)
        print(f"  R@5 diff:         {(g['recall_at_5']-t['recall_at_5'])*100:+.1f}%", flush=True)
        print(f"  Previous SOTA:    R@5=0.966", flush=True)
        print(f"  Our best:         R@5={max(t['recall_at_5'],g['recall_at_5']):.3f}", flush=True)

    # Also include v1 results for full picture
    v1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_longmemeval.json")
    if os.path.exists(v1_path):
        with open(v1_path) as f:
            v1 = json.load(f)
        print(f"\n  v1 whole-session Top-K: R@5={v1['topk']['recall_at_5']:.3f}", flush=True)
        print(f"  v1 whole-session Graph: R@5={v1['graph']['recall_at_5']:.3f}", flush=True)
    
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_longmemeval_v2.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}", flush=True)


if __name__ == "__main__":
    main()
