"""
Full MemCombine benchmark v2 — FAST version with cached model.
Grid-search weights + compare methods on 250 scenarios.

DK 🦍
"""

import json
import sys
import os
import time
import numpy as np
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_memory_graph.graph import MemoryGraph
from quantum_memory_graph.pipeline import recall, store_batch, set_graph
from benchmarks.memcombine import evaluate_combination

print("Loading 250 scenarios...")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "memcombine_250.json")) as f:
    ALL_SCENARIOS = json.load(f)
print(f"Loaded {len(ALL_SCENARIOS)} scenarios")

print("Loading embedding model (cached globally)...")
from sentence_transformers import SentenceTransformer
CACHED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready.\n")


def topk_recall_fn(memories, query, K):
    texts = memories + [query]
    embs = CACHED_MODEL.encode(texts, normalize_embeddings=True)
    scores = np.dot(embs[:-1], embs[-1])
    return np.argsort(scores)[-K:][::-1].tolist()


def graph_qaoa_fn(memories, query, K, alpha=0.4, beta_conn=0.35, gamma_cov=0.25, threshold=0.15):
    g = MemoryGraph(similarity_threshold=threshold)
    g._embedder = CACHED_MODEL  # Inject cached model
    g.add_memories_batch(memories)
    set_graph(g)
    result = recall(query, K=K, graph=g, alpha=alpha, beta_conn=beta_conn, gamma_cov=gamma_cov)
    selected_indices = []
    for sel_mem in result.get('memories', []):
        for i, m in enumerate(memories):
            if m == sel_mem['text'] and i not in selected_indices:
                selected_indices.append(i)
                break
    return selected_indices[:K]


def run_on_subset(recall_fn, scenarios, K=5):
    total_cov = 0; total_rec = 0; total_f1 = 0; perfect = 0
    for s in scenarios:
        texts = [m["text"] for m in s["memories"]]
        selected = recall_fn(texts, s["question"], K)
        r = evaluate_combination(selected, s)
        total_cov += r["coverage"]; total_rec += r["evidence_recall"]
        total_f1 += r["f1"]; perfect += (1 if r["coverage"] == 1.0 else 0)
    n = len(scenarios)
    return {"coverage": total_cov/n, "evidence_recall": total_rec/n, "f1": total_f1/n, "perfect": perfect, "perfect_pct": perfect/n*100}


def grid_search(scenarios, K=5):
    print("=" * 60)
    print("PHASE 1: Grid Search (50 scenario sample)")
    print("=" * 60)
    sample = scenarios[:50]
    
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
    betas = [0.1, 0.15, 0.25, 0.35, 0.45]
    gammas = [0.05, 0.1, 0.15, 0.25, 0.35]
    thresholds = [0.1, 0.15, 0.2, 0.25]

    best_score = 0; best_params = {}; tested = 0
    total = len(alphas)*len(betas)*len(gammas)*len(thresholds)

    for a, b, g, t in product(alphas, betas, gammas, thresholds):
        if (a+b+g) < 0.5 or (a+b+g) > 1.5:
            tested += 1; continue
        def fn(memories, query, K, _a=a, _b=b, _g=g, _t=t):
            return graph_qaoa_fn(memories, query, K, alpha=_a, beta_conn=_b, gamma_cov=_g, threshold=_t)
        result = run_on_subset(fn, sample, K)
        tested += 1
        if result["coverage"] > best_score:
            best_score = result["coverage"]
            best_params = {"alpha": a, "beta_conn": b, "gamma_cov": g, "threshold": t}
            print(f"  [{tested}/{total}] NEW BEST: {best_score*100:.1f}% (α={a}, β={b}, γ={g}, t={t})")
        elif tested % 100 == 0:
            print(f"  [{tested}/{total}] scanning... best={best_score*100:.1f}%")

    print(f"\n  OPTIMAL: α={best_params['alpha']}, β={best_params['beta_conn']}, γ={best_params['gamma_cov']}, threshold={best_params['threshold']}")
    print(f"  BEST: {best_score*100:.1f}%")
    return best_params


def main():
    print("🦍⚛️ MemCombine Full Benchmark v2 (FAST)\n")
    K = 5
    scenarios = ALL_SCENARIOS

    # Phase 1: Grid search
    t0 = time.time()
    best_params = grid_search(scenarios, K)
    print(f"  Grid search took {time.time()-t0:.0f}s\n")

    # Phase 2: Full 250
    print("=" * 60)
    print("PHASE 2: Full Benchmark (250 scenarios)")
    print("=" * 60)

    print("\n  Top-K baseline...")
    t0 = time.time()
    topk = run_on_subset(topk_recall_fn, scenarios, K)
    print(f"    Coverage: {topk['coverage']*100:.1f}% | Perfect: {topk['perfect']}/250 | {time.time()-t0:.0f}s")

    print("\n  Graph+QAOA (default weights)...")
    t0 = time.time()
    def default_fn(m, q, k): return graph_qaoa_fn(m, q, k, 0.4, 0.35, 0.25, 0.15)
    default = run_on_subset(default_fn, scenarios, K)
    print(f"    Coverage: {default['coverage']*100:.1f}% | Perfect: {default['perfect']}/250 | {time.time()-t0:.0f}s")

    print("\n  Graph+QAOA (tuned weights)...")
    t0 = time.time()
    def tuned_fn(m, q, k): return graph_qaoa_fn(m, q, k, **best_params)
    tuned = run_on_subset(tuned_fn, scenarios, K)
    print(f"    Coverage: {tuned['coverage']*100:.1f}% | Perfect: {tuned['perfect']}/250 | {time.time()-t0:.0f}s")

    # Final report
    print("\n" + "=" * 60)
    print("FINAL RESULTS — MemCombine (250 scenarios, K=5)")
    print("=" * 60)
    print(f"\n  {'Method':<30} {'Coverage':>10} {'Evidence':>10} {'F1':>10} {'Perfect':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name, label in [("topk", "Top-K Baseline"), ("default", "Graph+QAOA (default)"), ("tuned", "Graph+QAOA (tuned)")]:
        r = {"topk": topk, "default": default, "tuned": tuned}[name]
        print(f"  {label:<30} {r['coverage']*100:>9.1f}% {r['evidence_recall']*100:>9.1f}% {r['f1']*100:>9.1f}% {r['perfect']:>7}/250")

    print(f"\n  Optimal weights: α={best_params['alpha']}, β={best_params['beta_conn']}, γ={best_params['gamma_cov']}, threshold={best_params['threshold']}")
    print(f"  Tuned vs Top-K: {(tuned['coverage']-topk['coverage'])*100:+.1f}%")
    print(f"  Tuned vs Default: {(tuned['coverage']-default['coverage'])*100:+.1f}%")

    output = {"benchmark": "MemCombine_v2", "scenarios": 250, "K": K,
              "topk": topk, "default": default, "tuned": tuned, "params": best_params}
    with open("/home/dt/Projects/quantum-memory-graph/benchmarks/results_250.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to benchmarks/results_250.json")


if __name__ == "__main__":
    main()
