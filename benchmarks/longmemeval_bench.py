"""
LongMemEval Benchmark for quantum-memory-graph.
Runs the same 500-question benchmark MemPalace uses.

Each question has ~53 conversation sessions as haystack.
Task: retrieve the correct session(s) in top-K results.
Metric: Recall@5, Recall@10, NDCG@10.

DK 🦍
"""

import json
import sys
import os
import time
import math
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_memory_graph.graph import MemoryGraph

print("Loading embedding model (cached)...", flush=True)
from sentence_transformers import SentenceTransformer
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready.", flush=True)


def flatten_session(session):
    """Convert a session (list of turns) to a single text."""
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


def topk_recall(sessions_text, query, K):
    """Baseline: pure cosine similarity top-K."""
    texts = sessions_text + [query]
    embs = MODEL.encode(texts, normalize_embeddings=True)
    scores = np.dot(embs[:-1], embs[-1])
    ranked = np.argsort(scores)[::-1][:K].tolist()
    return ranked, scores


def graph_qaoa_recall(sessions_text, query, K, alpha=0.3, beta=0.25, gamma=0.15, threshold=0.15):
    """Our system: knowledge graph + QAOA subgraph selection."""
    g = MemoryGraph(similarity_threshold=threshold)
    g._embedder = MODEL
    g.add_memories_batch(sessions_text)
    
    from quantum_memory_graph.pipeline import recall as qrecall, set_graph
    set_graph(g)
    result = qrecall(query, K=K, graph=g, alpha=alpha, beta_conn=beta, gamma_cov=gamma)
    
    selected_indices = []
    for sel_mem in result.get('memories', []):
        for i, t in enumerate(sessions_text):
            if t == sel_mem['text'] and i not in selected_indices:
                selected_indices.append(i)
                break
    return selected_indices[:K]


def recall_at_k(retrieved_indices, gold_indices, K):
    """Recall@K: fraction of gold items found in top-K retrieved."""
    retrieved_set = set(retrieved_indices[:K])
    gold_set = set(gold_indices)
    if not gold_set:
        return 1.0
    return len(retrieved_set & gold_set) / len(gold_set)


def ndcg_at_k(retrieved_indices, gold_indices, K):
    """NDCG@K: normalized discounted cumulative gain."""
    gold_set = set(gold_indices)
    if not gold_set:
        return 1.0
    
    dcg = 0.0
    for i, idx in enumerate(retrieved_indices[:K]):
        if idx in gold_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1)=0
    
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), K)))
    return dcg / idcg if idcg > 0 else 0.0


def run_benchmark(data, method="topk", K=5, limit=None, alpha=0.3, beta=0.25, gamma=0.15, threshold=0.15):
    """Run benchmark on all questions."""
    if limit:
        data = data[:limit]
    
    recall5_total = 0; recall10_total = 0; ndcg10_total = 0
    n = len(data)
    
    for i, item in enumerate(data):
        question = item["question"]
        sessions = item["haystack_sessions"]
        gold_session_ids = item["answer_session_ids"]
        all_session_ids = item["haystack_session_ids"]
        
        # Flatten sessions to text
        sessions_text = [flatten_session(s) for s in sessions]
        
        # Map gold session IDs to indices
        gold_indices = []
        for gid in gold_session_ids:
            if gid in all_session_ids:
                gold_indices.append(all_session_ids.index(gid))
        
        if method == "topk":
            retrieved, _ = topk_recall(sessions_text, question, max(K, 10))
        else:
            retrieved = graph_qaoa_recall(sessions_text, question, max(K, 10),
                                          alpha=alpha, beta=beta, gamma=gamma, threshold=threshold)
        
        r5 = recall_at_k(retrieved, gold_indices, 5)
        r10 = recall_at_k(retrieved, gold_indices, 10)
        n10 = ndcg_at_k(retrieved, gold_indices, 10)
        
        recall5_total += r5
        recall10_total += r10
        ndcg10_total += n10
        
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{n}] R@5={recall5_total/(i+1):.3f} R@10={recall10_total/(i+1):.3f} NDCG@10={ndcg10_total/(i+1):.3f}", flush=True)
    
    return {
        "recall_at_5": recall5_total / n,
        "recall_at_10": recall10_total / n,
        "ndcg_at_10": ndcg10_total / n,
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser(description="LongMemEval benchmark")
    parser.add_argument("data", help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions")
    parser.add_argument("--method", choices=["topk", "graph", "both"], default="both")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()
    
    print(f"Loading LongMemEval data...", flush=True)
    with open(args.data) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions", flush=True)
    
    results = {}
    
    if args.method in ("topk", "both"):
        print(f"\n{'='*60}", flush=True)
        print(f"Top-K Baseline (cosine similarity)", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()
        results["topk"] = run_benchmark(data, method="topk", limit=args.limit)
        elapsed = time.time() - t0
        r = results["topk"]
        print(f"\n  Recall@5:  {r['recall_at_5']:.3f}", flush=True)
        print(f"  Recall@10: {r['recall_at_10']:.3f}", flush=True)
        print(f"  NDCG@10:   {r['ndcg_at_10']:.3f}", flush=True)
        print(f"  Time: {elapsed:.0f}s", flush=True)
    
    if args.method in ("graph", "both"):
        print(f"\n{'='*60}", flush=True)
        print(f"Graph+QAOA (α={args.alpha}, β={args.beta}, γ={args.gamma}, t={args.threshold})", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()
        results["graph"] = run_benchmark(data, method="graph", limit=args.limit,
                                          alpha=args.alpha, beta=args.beta,
                                          gamma=args.gamma, threshold=args.threshold)
        elapsed = time.time() - t0
        r = results["graph"]
        print(f"\n  Recall@5:  {r['recall_at_5']:.3f}", flush=True)
        print(f"  Recall@10: {r['recall_at_10']:.3f}", flush=True)
        print(f"  NDCG@10:   {r['ndcg_at_10']:.3f}", flush=True)
        print(f"  Time: {elapsed:.0f}s", flush=True)
    
    if "topk" in results and "graph" in results:
        print(f"\n{'='*60}", flush=True)
        print(f"COMPARISON", flush=True)
        print(f"{'='*60}", flush=True)
        diff5 = results["graph"]["recall_at_5"] - results["topk"]["recall_at_5"]
        diff10 = results["graph"]["recall_at_10"] - results["topk"]["recall_at_10"]
        print(f"  R@5  improvement: {diff5:+.3f} ({diff5*100:+.1f}%)", flush=True)
        print(f"  R@10 improvement: {diff10:+.3f} ({diff10*100:+.1f}%)", flush=True)
        print(f"\n  MemPalace raw mode: 0.966 R@5", flush=True)
        print(f"  Our Graph+QAOA:     {results['graph']['recall_at_5']:.3f} R@5", flush=True)
    
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_longmemeval.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}", flush=True)


if __name__ == "__main__":
    main()
