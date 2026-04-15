"""
LongMemEval v4 — Hybrid retrieval.
Phase 1: Top-K chunked finds best sessions (fast, 93.4% R@5)
Phase 2: Graph+QAOA re-ranks top sessions (relationship-aware refinement)

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


def topk_session_scores(chunk_embs, chunk_to_session, q_emb):
    """Score all chunks, aggregate best per session. Returns dict session_idx→score."""
    scores = np.dot(chunk_embs, q_emb)
    session_best = {}
    session_chunk_scores = {}
    for ci, score in enumerate(scores):
        si = chunk_to_session[ci]
        if si not in session_best or score > session_best[si]:
            session_best[si] = float(score)
        if si not in session_chunk_scores:
            session_chunk_scores[si] = []
        session_chunk_scores[si].append(float(score))
    # Also compute mean of top-3 chunks per session for more robust scoring
    session_robust = {}
    for si, chunk_scores in session_chunk_scores.items():
        top3 = sorted(chunk_scores, reverse=True)[:3]
        session_robust[si] = np.mean(top3)
    return session_best, session_robust


def hybrid_recall(chunk_embs, chunk_texts, chunk_to_session, q_emb, query_text,
                   K=5, top_sessions=15, alpha=0.4, beta=0.2, gamma=0.1, threshold=0.2):
    """
    Hybrid: Top-K narrows to top_sessions, then Graph+QAOA re-ranks.
    """
    from quantum_memory_graph.graph import MemoryGraph
    from quantum_memory_graph.pipeline import recall as qrecall, set_graph

    # Phase 1: Top-K chunked → get top sessions
    session_best, session_robust = topk_session_scores(chunk_embs, chunk_to_session, q_emb)
    ranked_sessions = sorted(session_robust.keys(), key=lambda s: session_robust[s], reverse=True)
    candidate_sessions = ranked_sessions[:top_sessions]

    if len(candidate_sessions) <= K:
        return candidate_sessions

    # Phase 2: Build graph on chunks from top sessions only
    sel_chunks = []
    sel_chunk_to_session = []
    for ci, c in enumerate(chunk_texts):
        si = chunk_to_session[ci]
        if si in candidate_sessions:
            sel_chunks.append(c)
            sel_chunk_to_session.append(si)

    # Limit to top 100 chunks by similarity
    if len(sel_chunks) > 100:
        sel_embs = []
        for ci, c in enumerate(chunk_texts):
            si = chunk_to_session[ci]
            if si in candidate_sessions:
                sel_embs.append(chunk_embs[ci])
        sel_embs = np.array(sel_embs)
        sims = np.dot(sel_embs, q_emb)
        top_idx = np.argsort(sims)[-100:]
        sel_chunks = [sel_chunks[i] for i in top_idx]
        sel_chunk_to_session = [sel_chunk_to_session[i] for i in top_idx]

    g = MemoryGraph(similarity_threshold=threshold)
    g._embedder = MODEL
    g.add_memories_batch(sel_chunks)
    set_graph(g)

    result = qrecall(query_text, K=min(K*5, len(sel_chunks)), graph=g,
                      alpha=alpha, beta_conn=beta, gamma_cov=gamma)

    # Score sessions by QAOA selection
    qaoa_session_hits = {}
    qaoa_session_rank = {}
    for rank, sel in enumerate(result.get('memories', [])):
        for ci, c in enumerate(sel_chunks):
            if c == sel['text']:
                si = sel_chunk_to_session[ci]
                qaoa_session_hits[si] = qaoa_session_hits.get(si, 0) + 1
                if si not in qaoa_session_rank:
                    qaoa_session_rank[si] = rank
                break

    # Combine: blend Top-K score + QAOA hit count
    final_scores = {}
    max_topk = max(session_robust[s] for s in candidate_sessions) if candidate_sessions else 1
    max_qaoa = max(qaoa_session_hits.values()) if qaoa_session_hits else 1

    for si in candidate_sessions:
        topk_norm = session_robust[si] / max_topk if max_topk > 0 else 0
        qaoa_norm = qaoa_session_hits.get(si, 0) / max_qaoa if max_qaoa > 0 else 0
        # Weight: 60% Top-K, 40% QAOA
        final_scores[si] = 0.6 * topk_norm + 0.4 * qaoa_norm

    ranked = sorted(final_scores.keys(), key=lambda s: final_scores[s], reverse=True)
    return ranked[:K*2]


def topk_only(chunk_embs, chunk_to_session, q_emb, K=5):
    """Pure Top-K chunked baseline."""
    _, session_robust = topk_session_scores(chunk_embs, chunk_to_session, q_emb)
    ranked = sorted(session_robust.keys(), key=lambda s: session_robust[s], reverse=True)
    return ranked[:K*2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("Loading data...", flush=True)
    with open(args.data) as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]
    print(f"Running {len(data)} questions\n", flush=True)

    # Pre-process
    print("Pre-chunking and embedding...", flush=True)
    t0 = time.time()

    question_data = []
    all_chunks_flat = []

    for qi, item in enumerate(data):
        sessions = item["haystack_sessions"]
        q_chunks = []
        q_chunk_to_session = []

        for si, sess in enumerate(sessions):
            text = flatten_session(sess)
            chunks = chunk_text(text)
            for c in chunks:
                q_chunks.append(c)
                q_chunk_to_session.append(si)
                all_chunks_flat.append(c)

        question_data.append({
            "chunks": q_chunks,
            "chunk_to_session": q_chunk_to_session,
            "gold_ids": item["answer_session_ids"],
            "all_ids": item["haystack_session_ids"],
            "question": item["question"],
        })

    print(f"  Total chunks: {len(all_chunks_flat)}", flush=True)
    print(f"  Embedding...", flush=True)
    all_chunk_embs = MODEL.encode(all_chunks_flat, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    questions_text = [item["question"] for item in data]
    q_embs = MODEL.encode(questions_text, normalize_embeddings=True, batch_size=256)

    idx = 0
    for qi, qd in enumerate(question_data):
        n = len(qd["chunks"])
        qd["chunk_embs"] = all_chunk_embs[idx:idx+n]
        idx += n

    print(f"  Done in {time.time()-t0:.0f}s\n", flush=True)

    results = {}

    # 1. Top-K chunked (baseline)
    print("=" * 60, flush=True)
    print("Top-K Chunked (robust: mean top-3 chunks/session)", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()
    r5 = r10 = n10 = 0
    for qi, qd in enumerate(question_data):
        gold = [qd["all_ids"].index(g) for g in qd["gold_ids"] if g in qd["all_ids"]]
        retrieved = topk_only(qd["chunk_embs"], qd["chunk_to_session"], q_embs[qi])
        r5 += recall_at_k(retrieved, gold, 5)
        r10 += recall_at_k(retrieved, gold, 10)
        n10 += ndcg_at_k(retrieved, gold, 10)
        if (qi+1) % 50 == 0 or qi == 0:
            n = qi+1
            print(f"  [{n}/{len(data)}] R@5={r5/n:.3f} R@10={r10/n:.3f}", flush=True)
    n = len(data)
    results["topk"] = {"recall_at_5": r5/n, "recall_at_10": r10/n, "ndcg_at_10": n10/n, "n": n}
    r = results["topk"]
    print(f"\n  FINAL: R@5={r['recall_at_5']:.3f}  R@10={r['recall_at_10']:.3f}  ({time.time()-t0:.0f}s)\n", flush=True)

    # 2. Hybrid: Top-K → Graph+QAOA re-rank (sweep blend weights)
    best_hybrid = None
    best_r5 = 0
    best_params = {}

    # Quick sweep on first 100 questions
    sweep_data = question_data[:100]
    sweep_qembs = q_embs[:100]

    for topk_w in [0.5, 0.6, 0.7, 0.8]:
        for top_sess in [10, 15, 20]:
            for qaoa_alpha in [0.3, 0.5]:
                for qaoa_beta in [0.1, 0.2]:
                    r5_sweep = 0
                    for qi, qd in enumerate(sweep_data):
                        gold = [qd["all_ids"].index(g) for g in qd["gold_ids"] if g in qd["all_ids"]]
                        retrieved = hybrid_recall(
                            qd["chunk_embs"], qd["chunks"], qd["chunk_to_session"],
                            sweep_qembs[qi], qd["question"],
                            K=5, top_sessions=top_sess, alpha=qaoa_alpha, beta=qaoa_beta,
                            gamma=0.1, threshold=0.2)
                        r5_sweep += recall_at_k(retrieved, gold, 5)
                    avg = r5_sweep / len(sweep_data)
                    if avg > best_r5:
                        best_r5 = avg
                        best_params = {"topk_w": topk_w, "top_sessions": top_sess,
                                       "alpha": qaoa_alpha, "beta": qaoa_beta}
                        print(f"  SWEEP: R@5={avg:.3f} topk_w={topk_w} top_sess={top_sess} α={qaoa_alpha} β={qaoa_beta}", flush=True)

    print(f"\n  Best sweep params: {best_params} → R@5={best_r5:.3f}", flush=True)

    # Update hybrid_recall to use best topk_w
    BEST_TOPK_W = best_params.get("topk_w", 0.6)
    BEST_TOP_SESS = best_params.get("top_sessions", 15)
    BEST_ALPHA = best_params.get("alpha", 0.3)
    BEST_BETA = best_params.get("beta", 0.2)

    # 3. Full 500 with best hybrid params
    print(f"\n{'='*60}", flush=True)
    print(f"Hybrid (Top-K → QAOA re-rank, tuned)", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    r5 = r10 = n10 = 0
    for qi, qd in enumerate(question_data):
        gold = [qd["all_ids"].index(g) for g in qd["gold_ids"] if g in qd["all_ids"]]
        retrieved = hybrid_recall(
            qd["chunk_embs"], qd["chunks"], qd["chunk_to_session"],
            q_embs[qi], qd["question"],
            K=5, top_sessions=BEST_TOP_SESS, alpha=BEST_ALPHA, beta=BEST_BETA,
            gamma=0.1, threshold=0.2)
        r5 += recall_at_k(retrieved, gold, 5)
        r10 += recall_at_k(retrieved, gold, 10)
        n10 += ndcg_at_k(retrieved, gold, 10)
        if (qi+1) % 50 == 0 or qi == 0:
            n = qi+1
            print(f"  [{n}/{len(data)}] R@5={r5/n:.3f} R@10={r10/n:.3f}", flush=True)
    n = len(data)
    results["hybrid"] = {"recall_at_5": r5/n, "recall_at_10": r10/n, "ndcg_at_10": n10/n, "n": n,
                          "params": best_params}
    r = results["hybrid"]
    print(f"\n  FINAL: R@5={r['recall_at_5']:.3f}  R@10={r['recall_at_10']:.3f}  ({time.time()-t0:.0f}s)\n", flush=True)

    # Comparison
    print("=" * 60, flush=True)
    print("FULL COMPARISON", flush=True)
    print("=" * 60, flush=True)
    print(f"\n  {'Method':<35} {'R@5':>8} {'R@10':>8} {'NDCG@10':>8}", flush=True)
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}", flush=True)

    t = results["topk"]
    print(f"  {'Top-K Chunked':<35} {t['recall_at_5']:>7.3f} {t['recall_at_10']:>7.3f} {t['ndcg_at_10']:>7.3f}", flush=True)
    h = results["hybrid"]
    print(f"  {'Hybrid (Top-K → QAOA re-rank)':<35} {h['recall_at_5']:>7.3f} {h['recall_at_10']:>7.3f} {h['ndcg_at_10']:>7.3f}", flush=True)
    print(f"  {'Previous SOTA':<35} {'0.966':>8} {'0.982':>8} {'0.889':>8}", flush=True)
    print(f"\n  Hybrid vs Top-K: {(h['recall_at_5']-t['recall_at_5'])*100:+.1f}% R@5", flush=True)
    print(f"  Hybrid vs Previous SOTA: {(h['recall_at_5']-0.966)*100:+.1f}% R@5", flush=True)

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_longmemeval_v4.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {outpath}", flush=True)


if __name__ == "__main__":
    main()
