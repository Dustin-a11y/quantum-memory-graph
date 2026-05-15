"""
Fast LongMemEval benchmark — caches session embeddings across all 500 questions.
Pre-computes all embeddings once, then tests Top-K vs Graph+QAOA per question.

DK 🦍
"""

import json, sys, os, time, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from quantum_memory_graph.graph import MemoryGraph
from quantum_memory_graph.pipeline import recall, set_graph, store_batch


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


def recall_at_k(retrieved, gold, K):
    return len(set(retrieved[:K]) & set(gold)) / len(gold) if gold else 1.0

def ndcg_at_k(retrieved, gold, K):
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    dcg = sum(1.0 / math.log2(i+2) for i, idx in enumerate(retrieved[:K]) if idx in gold_set)
    idcg = sum(1.0 / math.log2(i+2) for i in range(min(len(gold_set), K)))
    return dcg / idcg if idcg > 0 else 0.0


def run_benchmark(data_path, limit=None, K=10):
    print("Loading model (all-MiniLM-L6-v2)...", flush=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Loading data from {data_path}...", flush=True)
    with open(data_path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    n = len(data)
    print(f"Loaded {n} questions", flush=True)
    
    # Phase 1: Pre-compute all session embeddings (one massive batch)
    print("\nPre-computing session embeddings...", flush=True)
    all_sessions = []
    question_data = []
    
    for qi, item in enumerate(data):
        sessions_text = [flatten_session(s) for s in item["haystack_sessions"]]
        session_ids = item["haystack_session_ids"]
        gold_indices = [session_ids.index(g) for g in item["answer_session_ids"] if g in session_ids]
        
        all_sessions.extend(sessions_text)
        question_data.append({
            "question": item["question"],
            "sessions": sessions_text,
            "session_ids": session_ids,
            "gold": gold_indices,
            "n_sessions": len(sessions_text),
            "offset": len(all_sessions) - len(sessions_text),
        })
    
    t0 = time.time()
    all_embs = model.encode(all_sessions, normalize_embeddings=True, show_progress_bar=True)
    print(f"  Embedded {len(all_sessions)} sessions in {time.time()-t0:.0f}s", flush=True)
    
    # Pre-compute query embeddings
    print("Embedding queries...", flush=True)
    q_texts = [qd["question"] for qd in question_data]
    q_embs = model.encode(q_texts, normalize_embeddings=True, show_progress_bar=True)
    
    # Phase 2: Top-K baseline (pure cosine similarity)
    print("\n===== Top-K Baseline =====", flush=True)
    t0 = time.time()
    topk_r5 = topk_r10 = topk_n10 = 0.0
    
    for qi, qd in enumerate(question_data):
        offset = qd["offset"]
        n_sess = qd["n_sessions"]
        scores = np.dot(all_embs[offset:offset+n_sess], q_embs[qi])
        ranked = np.argsort(scores)[::-1].tolist()
        
        topk_r5 += recall_at_k(ranked, qd["gold"], 5)
        topk_r10 += recall_at_k(ranked, qd["gold"], 10)
        topk_n10 += ndcg_at_k(ranked, qd["gold"], 10)
        
        if (qi+1) % 100 == 0:
            print(f"  [{qi+1}/{n}] R@5={topk_r5/(qi+1):.3f} R@10={topk_r10/(qi+1):.3f}", flush=True)
    
    topk_time = time.time() - t0
    print(f"\n  Top-K Final: R@5={topk_r5/n:.3f} R@10={topk_r10/n:.3f} NDCG={topk_n10/n:.3f} ({topk_time:.0f}s)", flush=True)
    
    # Phase 3: Graph+QAOA
    print("\n===== Graph+QAOA (α=0.3, β=0.25, γ=0.15) =====", flush=True)
    t0 = time.time()
    graph_r5 = graph_r10 = graph_n10 = 0.0
    
    for qi, qd in enumerate(question_data):
        offset = qd["offset"]
        n_sess = qd["n_sessions"]
        
        # Build graph from pre-computed embeddings
        g = MemoryGraph(similarity_threshold=0.15)
        g._embedder = model
        
        for i in range(n_sess):
            mem_id = g._make_id(qd["sessions"][i])
            g.memories[mem_id] = g._make_memory(qd["sessions"][i], all_embs[offset+i])
            g.graph.add_node(mem_id)
        
        # Connect all memories
        mem_ids = list(g.memories.keys())
        for i in range(len(mem_ids)):
            for j in range(i+1, len(mem_ids)):
                weight = float(np.dot(all_embs[offset+i], all_embs[offset+j]))
                if weight > 0.15:
                    g.graph.add_edge(mem_ids[i], mem_ids[j], weight=weight, types=["semantic"])
        
        # Recall
        set_graph(g)
        result = recall(qd["question"], K=K, graph=g, alpha=0.3, beta_conn=0.25, gamma_cov=0.15)
        
        # Map result back to session indices
        selected = []
        for m in result.get("memories", []):
            for si, sess_text in enumerate(qd["sessions"]):
                if m["text"] == sess_text and si not in selected:
                    selected.append(si)
                    break
        
        graph_r5 += recall_at_k(selected, qd["gold"], 5)
        graph_r10 += recall_at_k(selected, qd["gold"], 10)
        graph_n10 += ndcg_at_k(selected, qd["gold"], 10)
        
        if (qi+1) % 100 == 0:
            n_completed_sec = max(1, time.time() - t0)
            rate = (qi+1) / n_completed_sec
            eta = (n - qi - 1) / rate
            print(f"  [{qi+1}/{n}] R@5={graph_r5/(qi+1):.3f} R@10={graph_r10/(qi+1):.3f} [{rate:.1f}q/s, ETA {eta:.0f}s]", flush=True)
    
    graph_time = time.time() - t0
    print(f"\n  Graph+QAOA Final: R@5={graph_r5/n:.3f} R@10={graph_r10/n:.3f} NDCG={graph_n10/n:.3f} ({graph_time:.0f}s)", flush=True)
    
    # Final comparison
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL COMPARISON ({n} questions)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\n  {'Method':<30} {'R@5':>8} {'R@10':>8} {'NDCG':>8} {'Time':>8}", flush=True)
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}", flush=True)
    print(f"  {'Top-K Baseline':<30} {topk_r5/n:>7.3f} {topk_r10/n:>7.3f} {topk_n10/n:>7.3f} {topk_time:>7.0f}s", flush=True)
    print(f"  {'Graph+QAOA':<30} {graph_r5/n:>7.3f} {graph_r10/n:>7.3f} {graph_n10/n:>7.3f} {graph_time:>7.0f}s", flush=True)
    
    advantage_r5 = (graph_r5 - topk_r5) / n * 100
    print(f"\n  QAOA Advantage: R@5 {advantage_r5:+.1f}%", flush=True)
    
    # Log results
    try:
        from benchmarks.data_collector import QMGBenchmarkLogger
        logger = QMGBenchmarkLogger()
        logger.log_longmemeval_run("topk", {
            "recall_at_5": topk_r5/n, "recall_at_10": topk_r10/n, "ndcg_at_10": topk_n10/n, "n": n
        }, "all-MiniLM-L6-v2", {"method": "embedding_only", "session_count": len(all_sessions)})
        logger.log_longmemeval_run("graph_qaoa", {
            "recall_at_5": graph_r5/n, "recall_at_10": graph_r10/n, "ndcg_at_10": graph_n10/n, "n": n
        }, "all-MiniLM-L6-v2", {"method": "graph_qaoa", "alpha": 0.3, "beta_conn": 0.25, "gamma_cov": 0.15, "threshold": 0.15})
        print(f"\n  Results logged to QMG benchmark tracker", flush=True)
    except Exception as e:
        print(f"\n  (Logging skipped: {e})", flush=True)
    
    print(f"\nDone. 🦍", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fast_longmemeval.py <data.json> [--limit N]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx+1])
    
    run_benchmark(data_path, limit=limit)
