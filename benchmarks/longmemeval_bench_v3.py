"""
LongMemEval v3 — Optimized chunking. Pre-embed all chunks once,
then score per question. Much faster than v2.

DK 🦍
"""

import json, sys, os, time, math, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading model...", flush=True)
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


def run_question_topk_chunked(chunk_embs, chunk_to_session, q_emb, K):
    """Score chunks against query, aggregate best per session."""
    scores = np.dot(chunk_embs, q_emb)
    session_best = {}
    for ci, score in enumerate(scores):
        si = chunk_to_session[ci]
        if si not in session_best or score > session_best[si]:
            session_best[si] = score
    ranked = sorted(session_best.keys(), key=lambda s: session_best[s], reverse=True)
    return ranked[:K*2]


def run_question_graph_chunked(chunk_embs, chunk_texts, chunk_to_session, q_emb, K,
                                query_text="",
                                alpha=0.3, beta=0.25, gamma=0.15, threshold=0.15):
    """Pre-filter top chunks, build graph, QAOA select, map to sessions."""
    from quantum_memory_graph.graph import MemoryGraph
    from quantum_memory_graph.pipeline import recall as qrecall, set_graph

    scores = np.dot(chunk_embs, q_emb)
    # Take top 80 chunks (balances speed vs coverage)
    top_n = min(80, len(scores))
    top_indices = np.argsort(scores)[-top_n:]
    
    sel_texts = [chunk_texts[i] for i in top_indices]
    sel_map = [chunk_to_session[i] for i in top_indices]
    
    g = MemoryGraph(similarity_threshold=threshold)
    g._embedder = MODEL
    g.add_memories_batch(sel_texts)
    set_graph(g)
    
    # Ask for more results to map back to sessions
    result = qrecall(query_text, K=min(K*5, top_n), graph=g, alpha=alpha, beta_conn=beta, gamma_cov=gamma)

    session_hits = {}
    session_rank = {}
    for rank, sel in enumerate(result.get('memories', [])):
        for ci, c in enumerate(sel_texts):
            if c == sel['text']:
                si = sel_map[ci]
                session_hits[si] = session_hits.get(si, 0) + 1
                if si not in session_rank:
                    session_rank[si] = rank
                break
    
    # If QAOA returned nothing useful, fall back to top-K
    if not session_hits:
        return run_question_topk_chunked(chunk_embs, chunk_to_session, q_emb, K)
    
    ranked = sorted(session_hits.keys(), key=lambda s: (-session_hits[s], session_rank.get(s, 999)))
    return ranked[:K*2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method", choices=["topk", "graph", "both"], default="both")
    args = parser.parse_args()
    
    print("Loading data...", flush=True)
    with open(args.data) as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]
    print(f"Running {len(data)} questions\n", flush=True)
    
    # Pre-process: chunk and embed ALL sessions for ALL questions
    # Since sessions repeat across questions, cache by session ID
    print("Pre-chunking and embedding all sessions...", flush=True)
    t0 = time.time()
    
    # Build per-question chunk data
    question_data = []
    all_chunks_flat = []  # For batch embedding
    chunk_positions = []  # (question_idx, local_chunk_idx)
    
    for qi, item in enumerate(data):
        sessions = item["haystack_sessions"]
        q_chunks = []
        q_chunk_to_session = []
        
        for si, sess in enumerate(sessions):
            text = flatten_session(sess)
            chunks = chunk_text(text)
            for c in chunks:
                local_idx = len(q_chunks)
                q_chunks.append(c)
                q_chunk_to_session.append(si)
                all_chunks_flat.append(c)
                chunk_positions.append((qi, local_idx))
        
        question_data.append({
            "chunks": q_chunks,
            "chunk_to_session": q_chunk_to_session,
            "gold_ids": item["answer_session_ids"],
            "all_ids": item["haystack_session_ids"],
            "question": item["question"],
        })
    
    print(f"  Total chunks: {len(all_chunks_flat)}", flush=True)
    print(f"  Avg chunks/question: {len(all_chunks_flat)/len(data):.0f}", flush=True)
    
    # Batch embed ALL chunks + ALL questions
    questions_text = [item["question"] for item in data]
    print(f"  Embedding {len(all_chunks_flat)} chunks...", flush=True)
    all_chunk_embs = MODEL.encode(all_chunks_flat, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    print(f"  Embedding {len(questions_text)} questions...", flush=True)
    q_embs = MODEL.encode(questions_text, normalize_embeddings=True, batch_size=256)
    
    # Distribute embeddings back to per-question arrays
    idx = 0
    for qi, qd in enumerate(question_data):
        n_chunks = len(qd["chunks"])
        qd["chunk_embs"] = all_chunk_embs[idx:idx+n_chunks]
        idx += n_chunks
    
    print(f"  Pre-processing done in {time.time()-t0:.0f}s\n", flush=True)
    
    results = {}
    
    # Top-K Chunked
    if args.method in ("topk", "both"):
        print("=" * 60, flush=True)
        print("Top-K Chunked", flush=True)
        print("=" * 60, flush=True)
        t0 = time.time()
        r5 = r10 = n10 = 0
        for qi, qd in enumerate(question_data):
            gold = [qd["all_ids"].index(g) for g in qd["gold_ids"] if g in qd["all_ids"]]
            retrieved = run_question_topk_chunked(qd["chunk_embs"], qd["chunk_to_session"], q_embs[qi], 5)
            r5 += recall_at_k(retrieved, gold, 5)
            r10 += recall_at_k(retrieved, gold, 10)
            n10 += ndcg_at_k(retrieved, gold, 10)
            if (qi+1) % 50 == 0 or qi == 0:
                n = qi+1
                print(f"  [{n}/{len(data)}] R@5={r5/n:.3f} R@10={r10/n:.3f} NDCG={n10/n:.3f}", flush=True)
        n = len(data)
        results["topk"] = {"recall_at_5": r5/n, "recall_at_10": r10/n, "ndcg_at_10": n10/n, "n": n}
        r = results["topk"]
        print(f"\n  FINAL: R@5={r['recall_at_5']:.3f}  R@10={r['recall_at_10']:.3f}  NDCG={r['ndcg_at_10']:.3f}  ({time.time()-t0:.0f}s)\n", flush=True)
    
    # Graph+QAOA Chunked
    if args.method in ("graph", "both"):
        print("=" * 60, flush=True)
        print("Graph+QAOA Chunked (top-80 → graph → QAOA)", flush=True)
        print("=" * 60, flush=True)
        t0 = time.time()
        r5 = r10 = n10 = 0
        for qi, qd in enumerate(question_data):
            gold = [qd["all_ids"].index(g) for g in qd["gold_ids"] if g in qd["all_ids"]]
            retrieved = run_question_graph_chunked(
                qd["chunk_embs"], qd["chunks"], qd["chunk_to_session"], q_embs[qi], 5,
                query_text=qd["question"])
            r5 += recall_at_k(retrieved, gold, 5)
            r10 += recall_at_k(retrieved, gold, 10)
            n10 += ndcg_at_k(retrieved, gold, 10)
            if (qi+1) % 50 == 0 or qi == 0:
                n = qi+1
                print(f"  [{n}/{len(data)}] R@5={r5/n:.3f} R@10={r10/n:.3f} NDCG={n10/n:.3f}", flush=True)
        n = len(data)
        results["graph"] = {"recall_at_5": r5/n, "recall_at_10": r10/n, "ndcg_at_10": n10/n, "n": n}
        r = results["graph"]
        print(f"\n  FINAL: R@5={r['recall_at_5']:.3f}  R@10={r['recall_at_10']:.3f}  NDCG={r['ndcg_at_10']:.3f}  ({time.time()-t0:.0f}s)\n", flush=True)
    
    # Comparison
    print("=" * 60, flush=True)
    print("FULL COMPARISON", flush=True)
    print("=" * 60, flush=True)
    print(f"\n  {'Method':<30} {'R@5':>8} {'R@10':>8} {'NDCG@10':>8}", flush=True)
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}", flush=True)
    
    # v1 results
    v1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_longmemeval.json")
    if os.path.exists(v1_path):
        with open(v1_path) as f:
            v1 = json.load(f)
        print(f"  {'v1 Top-K (whole session)':<30} {v1['topk']['recall_at_5']:>7.3f} {v1['topk']['recall_at_10']:>7.3f} {v1['topk']['ndcg_at_10']:>7.3f}", flush=True)
        print(f"  {'v1 Graph (whole session)':<30} {v1['graph']['recall_at_5']:>7.3f} {v1['graph']['recall_at_10']:>7.3f} {v1['graph']['ndcg_at_10']:>7.3f}", flush=True)
    
    if "topk" in results:
        r = results["topk"]
        print(f"  {'v3 Top-K (chunked)':<30} {r['recall_at_5']:>7.3f} {r['recall_at_10']:>7.3f} {r['ndcg_at_10']:>7.3f}", flush=True)
    if "graph" in results:
        r = results["graph"]
        print(f"  {'v3 Graph+QAOA (chunked)':<30} {r['recall_at_5']:>7.3f} {r['recall_at_10']:>7.3f} {r['ndcg_at_10']:>7.3f}", flush=True)
    
    print(f"  {'MemPalace raw':<30} {'0.966':>8} {'0.982':>8} {'0.889':>8}", flush=True)
    
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_longmemeval_v3.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {outpath}", flush=True)


if __name__ == "__main__":
    main()
