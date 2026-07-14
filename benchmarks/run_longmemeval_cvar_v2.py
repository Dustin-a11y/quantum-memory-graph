#!/usr/bin/env python3
"""
LongMemEval 500-question Benchmark — QMG CVaR subgraph optimizer.

Routes each question through the QMG subgraph optimizer on Spark.
Measures recall@K against gold answer sessions.

Usage:
    python3 -u run_longmemeval_cvar.py --limit 5    # Quick test
    python3 -u run_longmemeval_cvar.py --force       # Full 500
    python3 -u run_longmemeval_cvar.py --fast        # Skip QMG, cosine only

Output: JSON results + CSV saved to benchmarks/ directory.
"""
import json, time, math, sys, os, argparse, csv
from datetime import datetime, timezone
import numpy as np

DATA_PATH = os.environ.get("LONGMEMEVAL_DATA_PATH", "longmemeval_s_cleaned.json")
RESULTS_DIR = os.environ.get("QMG_BENCHMARK_RESULTS_DIR", os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(RESULTS_DIR, "longmemeval_cvar_results.json")
CSV_FILE = os.path.join(RESULTS_DIR, "longmemeval_cvar_results.csv")

T_START = time.time()

def flatten_session(session):
    if isinstance(session, str): return session
    if isinstance(session, list):
        parts = []
        for turn in session:
            if isinstance(turn, dict):
                parts.append(f"{turn.get('role','')}: {turn.get('content', turn.get('text', str(turn)))}")
            else:
                parts.append(str(turn))
        return "\n".join(parts)
    return str(session)

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
    parser.add_argument("--fast", action="store_true", help="Skip QMG, cosine only")
    parser.add_argument("--force", action="store_true", help="Run full 500")
    args = parser.parse_args()

    data = load_data(DATA_PATH)
    print(f"Loaded {len(data)} questions", flush=True)

    limit = args.limit
    if args.force: limit = None
    if limit: data = data[:limit]

    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading gte-large on {device}...", flush=True)
    model = SentenceTransformer("thenlper/gte-large", device=device)
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded, dim={dim}", flush=True)

    results = []
    n_questions = len(data)

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

        texts = [flatten_session(s) for s in haystack]

        # Encode
        t0 = time.time()
        all_texts = [question] + texts
        embs = model.encode(all_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        q_emb = embs[0]
        sess_embs = embs[1:]
        encode_time = time.time() - t0

        n_sessions = len(sess_embs)
        K_target = min(5, n_sessions)

        # Cosine baseline
        t0 = time.time()
        cos_scores = q_emb @ sess_embs.T
        cos_ranked = np.argsort(cos_scores)[::-1].tolist()
        cos_time = time.time() - t0

        r = {
            "idx": idx,
            "question": question[:120],
            "n_sessions": n_sessions,
            "n_gold": len(gold_indices),
            "cosine": {
                "r1": float(recall_at_k(cos_ranked, gold_indices, 1)),
                "r5": float(recall_at_k(cos_ranked, gold_indices, 5)),
                "r10": float(recall_at_k(cos_ranked, gold_indices, 10)),
                "ndcg": float(ndcg_at_k(cos_ranked, gold_indices, 10)),
                "time": cos_time,
            }
        }

        # QMG CVaR optimizer — two configs
        if not args.fast:
            t0 = time.time()
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from quantum_memory_graph.subgraph_optimizer import optimize_subgraph

                # Build adjacency from session embeddings (cosine similarity matrix)
                adj = sess_embs @ sess_embs.T
                np.fill_diagonal(adj, 0.0)

                for cfg_name, cfg in [
                    ("default", {"alpha": 0.4, "beta_conn": 0.35, "gamma_cov": 0.25, "shots": 4096}),
                    ("retrieval", {"alpha": 1.0, "beta_conn": 0.0, "gamma_cov": 0.0, "shots": 4096}),
                ]:
                    # Cap candidates at 14 for QAOA to avoid memory OOM
                    # (2^14 = 16K complex numbers, 2^40 = 17TB)
                    top_indices = np.argsort(cos_scores)[::-1][:14]
                    top_scores = cos_scores[top_indices]
                    top_adj = adj[np.ix_(top_indices, top_indices)]

                    result = optimize_subgraph(
                        relevance_scores=top_scores,
                        adjacency=top_adj,
                        K=K_target,
                        alpha=cfg["alpha"],
                        beta_conn=cfg["beta_conn"],
                        gamma_cov=cfg["gamma_cov"],
                        grid_size=6,
                        shots=cfg["shots"],
                        p_layers=2,
                    )
                    selection_raw = result.get("selection", [])
                    method = result.get("method", "unknown")

                    # Map capped indices back to original indices
                    selection = [top_indices[s] for s in selection_raw]

                    sel_set = set(selection)
                    ranked = list(selection)
                    for i in range(n_sessions):
                        if len(ranked) >= n_sessions: break
                        if i not in sel_set: ranked.append(i)

                    r[cfg_name] = {
                        "r1": float(recall_at_k(ranked, gold_indices, 1)),
                        "r5": float(recall_at_k(ranked, gold_indices, 5)),
                        "r10": float(recall_at_k(ranked, gold_indices, 10)),
                        "ndcg": float(ndcg_at_k(ranked, gold_indices, 10)),
                        "method": method,
                        "n_capped": len(top_indices),
                        "score": float(result.get("score", 0)),
                        "optimal_score": float(result.get("optimal", {}).get("score", 0)),
                        "time": time.time() - t0,
                    }

            except Exception as e:
                import traceback
                r["qmg_error"] = f"{type(e).__name__}: {e}"
                r["qmg_traceback"] = traceback.format_exc()

            r["total_qmg_time"] = time.time() - t0

        results.append(r)

        # Progress every 5 questions
        if (idx+1) % 5 == 0:
            elapsed = time.time() - T_START
            effective = [rr for rr in results if not rr.get("skip")]
            cos_done = [rr for rr in effective if "cosine" in rr]
            if cos_done:
                cos_r5_avg = np.mean([rr["cosine"]["r5"] for rr in cos_done]) * 100
                print(f"[{idx+1}/{n_questions}] {elapsed:.0f}s cos_r5={cos_r5_avg:.1f}%", flush=True)

    # Summary
    effective = [r for r in results if not r.get("skip")]

    cos_items = [r for r in effective if "cosine" in r]
    print("\n" + "="*60, flush=True)
    print(f"LONGMEMEVAL — {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"Questions: {len(effective)} effective ({len(results)-len(effective)} skipped)", flush=True)

    if cos_items:
        cos_r1 = np.mean([r["cosine"]["r1"] for r in cos_items])*100
        cos_r5 = np.mean([r["cosine"]["r5"] for r in cos_items])*100
        cos_r10 = np.mean([r["cosine"]["r10"] for r in cos_items])*100
        cos_ndcg = np.mean([r["cosine"]["ndcg"] for r in cos_items])
        print(f"\nCOSINE BASELINE:", flush=True)
        print(f"  R@1:  {cos_r1:.1f}%", flush=True)
        print(f"  R@5:  {cos_r5:.1f}%", flush=True)
        print(f"  R@10: {cos_r10:.1f}%", flush=True)
        print(f"  NDCG: {cos_ndcg:.4f}", flush=True)

    for cfg_name in ["default", "retrieval"]:
        items = [r for r in effective if cfg_name in r]
        if items:
            r1 = np.mean([r[cfg_name]["r1"] for r in items])*100
            r5 = np.mean([r[cfg_name]["r5"] for r in items])*100
            r10 = np.mean([r[cfg_name]["r10"] for r in items])*100
            ndcg = np.mean([r[cfg_name]["ndcg"] for r in items])
            methods = {}
            for r in items:
                m = r[cfg_name].get("method", "?")
                methods.setdefault(m, []).append(r[cfg_name]["r5"])
            avg_time = np.mean([r[cfg_name]["time"] for r in items])
            print(f"\nQMG {cfg_name.upper()}:", flush=True)
            print(f"  R@1:   {r1:.1f}%", flush=True)
            print(f"  R@5:   {r5:.1f}%", flush=True)
            print(f"  R@10:  {r10:.1f}%", flush=True)
            print(f"  NDCG:  {ndcg:.4f}", flush=True)
            print(f"  Avg time: {avg_time:.1f}s", flush=True)
            for m, vals in sorted(methods.items()):
                print(f"  {m}: {len(vals)}x R@5={np.mean(vals)*100:.1f}%", flush=True)

    total_t = time.time() - T_START
    print(f"\nTotal: {total_t:.0f}s ({total_t/60:.1f} min)", flush=True)
    print("="*60, flush=True)

    with open(RESULTS_FILE, "w") as f: json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "n_total": len(data), "results": results}, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_FILE}", flush=True)

    with open(CSV_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","n","ngold","cr1","cr5","cr10","cndcg",
                     "dr1","dr5","dr10","dndcg","dmethod",
                     "rr1","rr5","rr10","rndcg","rmethod"])
        for r in results:
            if r.get("skip"): continue
            w.writerow([
                r["idx"], r["n_sessions"], r["n_gold"],
                r["cosine"]["r1"], r["cosine"]["r5"], r["cosine"]["r10"], r["cosine"]["ndcg"],
                r.get("default", {}).get("r1"), r.get("default", {}).get("r5"),
                r.get("default", {}).get("r10"), r.get("default", {}).get("ndcg"),
                r.get("default", {}).get("method"),
                r.get("retrieval", {}).get("r1"), r.get("retrieval", {}).get("r5"),
                r.get("retrieval", {}).get("r10"), r.get("retrieval", {}).get("ndcg"),
                r.get("retrieval", {}).get("method"),
            ])
    print(f"CSV saved to {CSV_FILE}", flush=True)

if __name__ == "__main__":
    main()
