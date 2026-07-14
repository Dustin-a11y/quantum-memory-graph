#!/usr/bin/env python3
"""
LongMemEval 500 — Two-Stage Pipeline Benchmark.

Stage 1: gte-large cosine similarity -> candidate ranking
Stage 2: QAOA+CVaR subgraph refinement on top candidates

Measures: pure cosine vs cosine+QAOA refinement vs greedy subgraph

DK 🦍
"""
import json, time, math, sys, os, argparse, csv
from datetime import datetime, timezone
import numpy as np

DATA_PATH = os.environ.get("LONGMEMEVAL_DATA_PATH", "longmemeval_s_cleaned.json")
RESULTS_DIR = os.environ.get("QMG_BENCHMARK_RESULTS_DIR", os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(RESULTS_DIR, "longmemeval_staged_results.json")
CSV_FILE = os.path.join(RESULTS_DIR, "longmemeval_staged_results.csv")

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
    parser.add_argument("--max-candidates", type=int, default=14, help="QAOA candidate pool size")
    parser.add_argument("--top-k", type=int, default=5, help="Target selection K")
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

    # Trackers
    count_qaoa_won = 0
    count_greedy_won = 0
    count_tied = 0
    count_qaoa_runs = 0

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

        # --- Stage 1: Cosine ---
        t0 = time.time()
        cos_scores = q_emb @ sess_embs.T
        cos_ranking = np.argsort(cos_scores)[::-1].tolist()
        cos_time = time.time() - t0

        r = {
            "idx": idx,
            "question": question[:120],
            "n_sessions": n_sessions,
            "n_gold": len(gold_indices),
            "cosine": {
                "r1": float(recall_at_k(cos_ranking, gold_indices, 1)),
                "r5": float(recall_at_k(cos_ranking, gold_indices, 5)),
                "r10": float(recall_at_k(cos_ranking, gold_indices, 10)),
                "ndcg": float(ndcg_at_k(cos_ranking, gold_indices, 10)),
                "time": cos_time,
            },
        }

        # --- Stage 2: QAOA+CVaR refinement on top candidates ---
        try:
            t0 = time.time()
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from quantum_memory_graph.subgraph_optimizer import optimize_subgraph

            # Take candidates from cosine top-N
            top_indices = cos_ranking[:max_candidates]
            top_scores = cos_scores[top_indices]

            # Build adjacency from top-candidate embeddings
            top_embs = sess_embs[top_indices]
            adj = top_embs @ top_embs.T
            np.fill_diagonal(adj, 0.0)

            # Methods to compare
            for method_name, cfg in [
                ("qaoa_cvar", {"alpha": 0.4, "beta_conn": 0.35, "gamma_cov": 0.25, "shots": 4096, "p_layers": 2}),
                ("greedy_subgraph", {"alpha": 0.4, "beta_conn": 0.35, "gamma_cov": 0.25}),
            ]:
                result = optimize_subgraph(
                    relevance_scores=top_scores,
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

                # Map capped indices back
                selection = [top_indices[s] for s in selection_raw]

                # Build ranked list: QAOA picks first, then remaining in cosine order
                sel_set = set(selection)
                ranked = list(selection)
                for i in cos_ranking:
                    if len(ranked) >= n_sessions: break
                    if i not in sel_set:
                        ranked.append(i)

                r[method_name] = {
                    "r1": float(recall_at_k(ranked, gold_indices, 1)),
                    "r5": float(recall_at_k(ranked, gold_indices, 5)),
                    "r10": float(recall_at_k(ranked, gold_indices, 10)),
                    "ndcg": float(ndcg_at_k(ranked, gold_indices, 10)),
                    "method": opt_method,
                    "n_capped": len(top_indices),
                    "score": float(result.get("score", 0)),
                    "optimal_score": float(result.get("optimal", {}).get("score", 0)),
                    "time": time.time() - t0,
                }

            # Determine winner between QAOA and cosine
            q_r5 = r.get("qaoa_cvar", {}).get("r5", 0)
            c_r5 = r["cosine"]["r5"]
            if q_r5 > c_r5:
                r["stage2_winner"] = "qaoa_cvar"
                count_qaoa_won += 1
            elif c_r5 > q_r5:
                r["stage2_winner"] = "cosine"
                count_greedy_won += 1
            else:
                r["stage2_winner"] = "tie"
                count_tied += 1

            if r.get("qaoa_cvar", {}).get("method") == "qaoa":
                count_qaoa_runs += 1

        except Exception as e:
            import traceback
            r["stage2_error"] = "%s: %s" % (type(e).__name__, e)
            r["stage2_traceback"] = traceback.format_exc()

        results.append(r)

        # Progress
        if (idx+1) % 5 == 0:
            elapsed = time.time() - T_START
            effective = [rr for rr in results if not rr.get("skip")]
            if effective:
                c_r5_avg = np.mean([rr["cosine"]["r5"] for rr in effective]) * 100
                q_r5_avg = np.mean([rr.get("qaoa_cvar", {}).get("r5", 0) for rr in effective if "qaoa_cvar" in rr]) * 100
                q_wins = sum(1 for rr in effective if rr.get("stage2_winner") == "qaoa_cvar")
                print("[%d/%d] %.0fs | cos_r5=%.1f%% | qaoa_r5=%.1f%% | qaoa_wins=%d" % (
                    idx+1, n_questions, elapsed, c_r5_avg, q_r5_avg, q_wins), flush=True)

    # Summary
    effective = [r for r in results if not r.get("skip")]
    n_eff = len(effective)

    print("\n" + "=" * 80, flush=True)
    print("LONGMEMEVAL TWO-STAGE — %s" % datetime.now(timezone.utc).isoformat(), flush=True)
    print("Questions: %d effective (%d skipped)" % (n_eff, n_questions - n_eff), flush=True)
    print("Max candidates: %d, Target K: %d" % (max_candidates, top_k), flush=True)
    print()

    # Stage 1: Pure cosine
    cos_items = [r for r in effective if "cosine" in r]
    if cos_items:
        cos = {
            "r1": np.mean([r["cosine"]["r1"] for r in cos_items]) * 100,
            "r5": np.mean([r["cosine"]["r5"] for r in cos_items]) * 100,
            "r10": np.mean([r["cosine"]["r10"] for r in cos_items]) * 100,
            "ndcg": np.mean([r["cosine"]["ndcg"] for r in cos_items]),
        }
        print("--- STAGE 1: COSINE BASELINE ---")
        print("  R@1:  %.1f%%" % cos["r1"])
        print("  R@5:  %.1f%%" % cos["r5"])
        print("  R@10: %.1f%%" % cos["r10"])
        print("  NDCG: %.4f" % cos["ndcg"])
        print()

    # Stage 2: QAOA+CVaR refinement
    qaoa_items = [r for r in effective if "qaoa_cvar" in r]
    if qaoa_items:
        qaoa = {
            "r1": np.mean([r["qaoa_cvar"]["r1"] for r in qaoa_items]) * 100,
            "r5": np.mean([r["qaoa_cvar"]["r5"] for r in qaoa_items]) * 100,
            "r10": np.mean([r["qaoa_cvar"]["r10"] for r in qaoa_items]) * 100,
            "ndcg": np.mean([r["qaoa_cvar"]["ndcg"] for r in qaoa_items]),
        }
        print("--- STAGE 2: COSINE + QAOA REFINEMENT ---")
        print("  R@1:  %.1f%%" % qaoa["r1"])
        print("  R@5:  %.1f%%" % qaoa["r5"])
        print("  R@10: %.1f%%" % qaoa["r10"])
        print("  NDCG: %.4f" % qaoa["ndcg"])
        print()

    # Greedy subgraph baseline
    greedy_items = [r for r in effective if "greedy_subgraph" in r]
    if greedy_items:
        greedy = {
            "r1": np.mean([r["greedy_subgraph"]["r1"] for r in greedy_items]) * 100,
            "r5": np.mean([r["greedy_subgraph"]["r5"] for r in greedy_items]) * 100,
            "r10": np.mean([r["greedy_subgraph"]["r10"] for r in greedy_items]) * 100,
            "ndcg": np.mean([r["greedy_subgraph"]["ndcg"] for r in greedy_items]),
        }
        print("--- BASELINE: COSINE + GREEDY SUBGRAPH ---")
        print("  R@1:  %.1f%%" % greedy["r1"])
        print("  R@5:  %.1f%%" % greedy["r5"])
        print("  R@10: %.1f%%" % greedy["r10"])
        print("  NDCG: %.4f" % greedy["ndcg"])
        print()

    # Head-to-head: QAOA vs Cosine
    print("--- HEAD-TO-HEAD (QAOA refinement vs pure cosine) ---")
    print("  Questions where QAOA refinement WINS:  %d (%.1f%%)" % (count_qaoa_won, count_qaoa_won/n_eff*100))
    print("  Questions where cosine alone WINS:     %d (%.1f%%)" % (count_greedy_won, count_greedy_won/n_eff*100))
    print("  Ties:                                  %d (%.1f%%)" % (count_tied, count_tied/n_eff*100))
    print("  QAOA optimizer ran (%d/%d)" % (count_qaoa_runs, n_eff))
    print()

    # Delta vs baseline
    if qaoa_items and cos_items:
        delta_r1 = qaoa["r1"] - cos["r1"]
        delta_r5 = qaoa["r5"] - cos["r5"]
        delta_r10 = qaoa["r10"] - cos["r10"]
        print("--- DELTA (stage2 - stage1) ---")
        print("  R@1:  %+.1f%%" % delta_r1)
        print("  R@5:  %+.1f%%" % delta_r5)
        print("  R@10: %+.1f%%" % delta_r10)
        print()

    total_t = time.time() - T_START
    print("Total: %.0fs (%.1f min)" % (total_t, total_t/60), flush=True)
    print("=" * 80, flush=True)

    # Save JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_total": len(data),
            "config": {"max_candidates": max_candidates, "top_k": top_k},
            "cosine": cos if cos_items else {},
            "qaoa_cvar": qaoa if qaoa_items else {},
            "greedy_subgraph": greedy if greedy_items else {},
            "count_qaoa_won": count_qaoa_won,
            "count_cosine_won": count_greedy_won,
            "count_tied": count_tied,
            "count_qaoa_runs": count_qaoa_runs,
            "results": results,
        }, f, indent=2, default=str)
    print("Saved JSON to %s" % RESULTS_FILE, flush=True)

    # Save CSV
    with open(CSV_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx","n","ngold",
            "cr1","cr5","cr10","cndcg",
            "qr1","qr5","qr10","qndcg","qmethod",
            "gr1","gr5","gr10","gndcg","gmethod",
            "winner"
        ])
        for r in results:
            if r.get("skip"): continue
            def g(d, key): return d.get(key, "") if d else ""
            w.writerow([
                r["idx"], r["n_sessions"], r["n_gold"],
                g(r.get("cosine"), "r1"), g(r.get("cosine"), "r5"),
                g(r.get("cosine"), "r10"), g(r.get("cosine"), "ndcg"),
                g(r.get("qaoa_cvar"), "r1"), g(r.get("qaoa_cvar"), "r5"),
                g(r.get("qaoa_cvar"), "r10"), g(r.get("qaoa_cvar"), "ndcg"),
                g(r.get("qaoa_cvar"), "method"),
                g(r.get("greedy_subgraph"), "r1"), g(r.get("greedy_subgraph"), "r5"),
                g(r.get("greedy_subgraph"), "r10"), g(r.get("greedy_subgraph"), "ndcg"),
                g(r.get("greedy_subgraph"), "method"),
                r.get("stage2_winner", "?"),
            ])
    print("Saved CSV to %s" % CSV_FILE, flush=True)


if __name__ == "__main__":
    main()
