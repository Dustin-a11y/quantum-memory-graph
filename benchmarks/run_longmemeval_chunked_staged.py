#!/usr/bin/env python3
"""
LongMemEval 500 — True Two-Stage Pipeline.

Stage 1: Chunked gte-large embedding (v7-style) -> robust session scoring -> top-14
Stage 2: QAOA+CVaR subgraph refinement on top-14

DK 🦍
"""
import json, time, math, sys, os, argparse, csv
from datetime import datetime, timezone
import numpy as np

DATA_PATH = "/home/dt/projects-shared/LongMemEval/data/longmemeval_s_cleaned.json"
RESULTS_DIR = "/home/dt/qmg-v1/benchmarks"
RESULTS_FILE = os.path.join(RESULTS_DIR, "longmemeval_chunked_staged_results.json")
CSV_FILE = os.path.join(RESULTS_DIR, "longmemeval_chunked_staged_results.csv")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

T_START = time.time()

def flatten_session(session):
    if isinstance(session, str): return session
    if isinstance(session, list):
        parts = []
        for turn in session:
            if isinstance(turn, dict):
                parts.append("%s: %s" % (turn.get('role',''), turn.get('content', turn.get('text', str(turn)))))
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

def load_data(path, limit=None):
    with open(path) as f: data = json.load(f)
    if not isinstance(data, list):
        for k in ["data","questions","items","results"]:
            if k in data: data = data[k]; break
    if limit: data = data[:limit]
    return data

def recall_at_k(ranked, gold, K):
    gold_set = set(gold)
    if not gold_set: return 1.0
    return 1.0 if set(ranked[:K]) & gold_set else 0.0

def ndcg_at_k(ranked, gold, K):
    gold_set = set(gold)
    if not gold_set: return 1.0
    dcg = sum(1.0/math.log2(i+2) for i,idx in enumerate(ranked[:K]) if idx in gold_set)
    idcg = sum(1.0/math.log2(i+2) for i in range(min(len(gold_set), K)))
    return dcg/idcg if idcg>0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Run full 500")
    parser.add_argument("--max-candidates", type=int, default=14)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    data = load_data(DATA_PATH)
    print("Loaded %d questions" % len(data), flush=True)

    limit = args.limit
    if args.force: limit = None
    if limit: data = data[:limit]

    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading gte-large on %s..." % device, flush=True)
    model = SentenceTransformer("thenlper/gte-large", device=device)
    dim = model.get_sentence_embedding_dimension()
    print("Model loaded, dim=%d" % dim, flush=True)

    results = []
    n_questions = len(data)
    max_candidates = args.max_candidates
    top_k = args.top_k
    count_qaoa_won = 0
    count_cosine_won = 0
    count_tied = 0

    for idx, item in enumerate(data):
        question = item.get("question", item.get("query", ""))
        haystack = item.get("haystack_sessions", item.get("sessions", item.get("corpus", [])))
        haystack_ids = item.get("haystack_session_ids", item.get("session_ids", []))
        answer_ids = item.get("answer_session_ids", item.get("answer_ids", []))

        gold_indices = []
        for g in answer_ids:
            try: gold_indices.append(haystack_ids.index(g))
            except ValueError: pass

        if not gold_indices or len(haystack) < 3:
            results.append({"idx": idx, "skip": True, "reason": "no_gold_or_too_few"})
            continue

        # --- Stage 1: Chunk-level embedding ---
        t0 = time.time()
        n_sessions = len(haystack)

        # Chunk each session
        all_chunks = []
        chunk_to_session = []
        for si, sess in enumerate(haystack):
            text = flatten_session(sess)
            for c in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
                all_chunks.append(c)
                chunk_to_session.append(si)

        # Embed question and all chunks
        q_emb = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        chunk_embs = model.encode(all_chunks, normalize_embeddings=True, batch_size=256, show_progress_bar=False)

        # Score: for each session, take mean of top-3 chunk scores
        chunk_scores = chunk_embs @ q_emb
        chunk_scores_list = chunk_scores.tolist()

        session_chunk_scores = {}
        for ci, si in enumerate(chunk_to_session):
            if si not in session_chunk_scores:
                session_chunk_scores[si] = []
            session_chunk_scores[si].append(chunk_scores_list[ci])

        # Robust session score = mean of top 3 chunk scores (or all if < 3)
        session_scores = {}
        for si, scores in session_chunk_scores.items():
            sorted_scores = sorted(scores, reverse=True)
            top_n = min(3, len(sorted_scores))
            session_scores[si] = np.mean(sorted_scores[:top_n])

        # Ranking
        session_score_list = [session_scores.get(i, -1.0) for i in range(n_sessions)]
        chunk_cosine_ranking = sorted(range(n_sessions), key=lambda i: session_score_list[i], reverse=True)

        # Also compute plain-cosine (session-level, no chunking) for comparison
        sess_texts = [flatten_session(s) for s in haystack]
        sess_embs = model.encode(sess_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        cos_scores = q_emb @ sess_embs.T
        plain_cosine_ranking = np.argsort(cos_scores)[::-1].tolist()

        # Session-level embeddings for adjacency matrix
        sess_embs_full = model.encode(sess_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

        plain_cosine_time = time.time() - t0

        r = {
            "idx": idx,
            "question": question[:120],
            "n_sessions": n_sessions,
            "n_chunks": len(all_chunks),
            "n_gold": len(gold_indices),
            "time": plain_cosine_time,
            "chunked_cosine": {
                "r1": float(recall_at_k(chunk_cosine_ranking, gold_indices, 1)),
                "r5": float(recall_at_k(chunk_cosine_ranking, gold_indices, 5)),
                "r10": float(recall_at_k(chunk_cosine_ranking, gold_indices, 10)),
                "ndcg": float(ndcg_at_k(chunk_cosine_ranking, gold_indices, 10)),
            },
            "plain_cosine": {
                "r1": float(recall_at_k(plain_cosine_ranking, gold_indices, 1)),
                "r5": float(recall_at_k(plain_cosine_ranking, gold_indices, 5)),
                "r10": float(recall_at_k(plain_cosine_ranking, gold_indices, 10)),
                "ndcg": float(ndcg_at_k(plain_cosine_ranking, gold_indices, 10)),
            },
        }

        # --- Stage 2: QAOA+CVaR refinement on chunk-cosine top candidates ---
        try:
            t0 = time.time()
            sys.path.insert(0, "/home/dt/qmg-v1")
            from quantum_memory_graph.subgraph_optimizer import optimize_subgraph

            # Take top candidates from chunked cosine
            top_indices = chunk_cosine_ranking[:max_candidates]
            top_scores_vals = [session_score_list[i] for i in top_indices]

            # Build adjacency from session-level embeddings
            top_embs = sess_embs_full[top_indices]
            adj = top_embs @ top_embs.T
            np.fill_diagonal(adj, 0.0)

            for method_name, cfg in [
                ("qaoa_refined", {"alpha": 0.4, "beta_conn": 0.35, "gamma_cov": 0.25, "shots": 4096, "p_layers": 2}),
                ("greedy_refined", {"alpha": 0.4, "beta_conn": 0.35, "gamma_cov": 0.25}),
            ]:
                result = optimize_subgraph(
                    relevance_scores=np.array(top_scores_vals),
                    adjacency=adj,
                    K=top_k,
                    alpha=cfg["alpha"],
                    beta_conn=cfg["beta_conn"],
                    gamma_cov=cfg["gamma_cov"],
                    grid_size=6,
                    shots=cfg.get("shots", 4096),
                    p_layers=cfg.get("p_layers", 2),
                )
                selection_raw = result.get("selection", [])
                opt_method = result.get("method", "unknown")
                selection = [top_indices[s] for s in selection_raw]

                sel_set = set(selection)
                ranked = list(selection)
                for i in chunk_cosine_ranking:
                    if len(ranked) >= n_sessions: break
                    if i not in sel_set: ranked.append(i)

                r[method_name] = {
                    "r1": float(recall_at_k(ranked, gold_indices, 1)),
                    "r5": float(recall_at_k(ranked, gold_indices, 5)),
                    "r10": float(recall_at_k(ranked, gold_indices, 10)),
                    "ndcg": float(ndcg_at_k(ranked, gold_indices, 10)),
                    "method": opt_method,
                    "score": float(result.get("score", 0)),
                    "time": time.time() - t0,
                }

            # Head-to-head: chunked cosine vs QAOA refinement
            c_r5 = r["chunked_cosine"]["r5"]
            q_r5 = r.get("qaoa_refined", {}).get("r5", 0)
            if q_r5 > c_r5:
                r["winner"] = "qaoa"
                count_qaoa_won += 1
            elif c_r5 > q_r5:
                r["winner"] = "chunked_cosine"
                count_cosine_won += 1
            else:
                r["winner"] = "tie"
                count_tied += 1

        except Exception as e:
            import traceback
            r["stage2_error"] = "%s: %s" % (type(e).__name__, e)

        results.append(r)

        if (idx+1) % 5 == 0:
            elapsed = time.time() - T_START
            eff = [rr for rr in results if not rr.get("skip")]
            if eff:
                cc5 = np.mean([rr["chunked_cosine"]["r5"] for rr in eff]) * 100
                pc5 = np.mean([rr["plain_cosine"]["r5"] for rr in eff]) * 100
                qa5 = np.mean([rr.get("qaoa_refined",{}).get("r5",0) for rr in eff if "qaoa_refined" in rr]) * 100
                qw = sum(1 for rr in eff if rr.get("winner") == "qaoa")
                print("[%d/%d] %.0fs | chunk=%.1f%% plain=%.1f%% qaoa=%.1f%% wins=%d" % (
                    idx+1, n_questions, elapsed, cc5, pc5, qa5, qw), flush=True)

    # Summary
    eff = [r for r in results if not r.get("skip")]
    ne = len(eff)
    print("\n" + "=" * 80, flush=True)
    print("LONGMEMEVAL TWO-STAGE (CHUNKED) — %s" % datetime.now(timezone.utc).isoformat(), flush=True)
    print("Questions: %d | Chunk size=%d overlap=%d candidates=%d" % (ne, CHUNK_SIZE, CHUNK_OVERLAP, max_candidates), flush=True)
    print()

    for label, key in [("CHUNKED COSINE (v7-style)", "chunked_cosine"),
                        ("PLAIN COSINE (no chunking)", "plain_cosine"),
                        ("CHUNKED + QAOA REFINEMENT", "qaoa_refined"),
                        ("CHUNKED + GREEDY REFINEMENT", "greedy_refined")]:
        items = [r for r in eff if key in r]
        if items:
            r1 = np.mean([r[key]["r1"] for r in items]) * 100
            r5 = np.mean([r[key]["r5"] for r in items]) * 100
            r10 = np.mean([r[key]["r10"] for r in items]) * 100
            ndcg = np.mean([r[key]["ndcg"] for r in items])
            print("  %-35s R@1=%.1f%% R@5=%.1f%% R@10=%.1f%% NDCG=%.4f" % (label, r1, r5, r10, ndcg))
    print()

    print("--- HEAD-TO-HEAD ---")
    print("  QAOA refinement wins:  %d (%.1f%%)" % (count_qaoa_won, count_qaoa_won/max(ne,1)*100))
    print("  Chunked cosine wins:   %d (%.1f%%)" % (count_cosine_won, count_cosine_won/max(ne,1)*100))
    print("  Ties:                  %d (%.1f%%)" % (count_tied, count_tied/max(ne,1)*100))

    # Delta chunked vs plain
    cc = [r for r in eff if "chunked_cosine" in r]
    pc = [r for r in eff if "plain_cosine" in r]
    if cc and pc:
        d5 = (np.mean([r["chunked_cosine"]["r5"] for r in cc]) - np.mean([r["plain_cosine"]["r5"] for r in pc])) * 100
        d10 = (np.mean([r["chunked_cosine"]["r10"] for r in cc]) - np.mean([r["plain_cosine"]["r10"] for r in pc])) * 100
        print()
        print("--- CHUNKING DELTA vs PLAIN ---")
        print("  R@5 delta:  %+.1f%%" % d5)
        print("  R@10 delta: %+.1f%%" % d10)

    total_t = time.time() - T_START
    print("\nTotal: %.0fs (%.1f min)" % (total_t, total_t/60), flush=True)
    print("=" * 80, flush=True)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "config": {"chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP, "max_candidates": max_candidates}, "results": results}, f, indent=2, default=str)

    with open(CSV_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","n","ngold","nchunks",
                     "cc_r1","cc_r5","cc_r10","cc_ndcg",
                     "pc_r1","pc_r5","pc_r10","pc_ndcg",
                     "q_r1","q_r5","q_r10","q_ndcg","q_method",
                     "g_r1","g_r5","g_r10","g_ndcg","g_method",
                     "winner"])
        for r in results:
            if r.get("skip"): continue
            def g(d, k): return d.get(k, "") if d else ""
            w.writerow([
                r["idx"], r["n_sessions"], r["n_gold"], r.get("n_chunks", 0),
                g(r.get("chunked_cosine"), "r1"), g(r.get("chunked_cosine"), "r5"),
                g(r.get("chunked_cosine"), "r10"), g(r.get("chunked_cosine"), "ndcg"),
                g(r.get("plain_cosine"), "r1"), g(r.get("plain_cosine"), "r5"),
                g(r.get("plain_cosine"), "r10"), g(r.get("plain_cosine"), "ndcg"),
                g(r.get("qaoa_refined"), "r1"), g(r.get("qaoa_refined"), "r5"),
                g(r.get("qaoa_refined"), "r10"), g(r.get("qaoa_refined"), "ndcg"),
                g(r.get("qaoa_refined"), "method"),
                g(r.get("greedy_refined"), "r1"), g(r.get("greedy_refined"), "r5"),
                g(r.get("greedy_refined"), "r10"), g(r.get("greedy_refined"), "ndcg"),
                g(r.get("greedy_refined"), "method"),
                r.get("winner", "?"),
            ])

    print("Saved results.", flush=True)


if __name__ == "__main__":
    main()
