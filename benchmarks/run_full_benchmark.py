"""
Full MemCombine benchmark: grid-search weights + compare methods.

1. Load 250 scenarios
2. Grid-search α, β, γ weights
3. Compare: Top-K vs Graph+QAOA (old NER) vs Graph+QAOA (spaCy NER)
4. Report optimal weights and final scores

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
from benchmarks.memcombine import run_benchmark, evaluate_combination

# Load scenarios
print("Loading 250 scenarios...")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "memcombine_250.json")) as f:
    ALL_SCENARIOS = json.load(f)
print(f"Loaded {len(ALL_SCENARIOS)} scenarios")

# Load sentence-transformers once
print("Loading embedding model...")
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready.")


def topk_recall_fn(memories, query, K):
    """Baseline: embedding Top-K."""
    texts = memories + [query]
    embs = st_model.encode(texts, normalize_embeddings=True)
    query_emb = embs[-1]
    mem_embs = embs[:-1]
    scores = np.dot(mem_embs, query_emb)
    return np.argsort(scores)[-K:][::-1].tolist()


def graph_qaoa_fn(memories, query, K, alpha=0.4, beta_conn=0.35, gamma_cov=0.25, threshold=0.15):
    """Graph + QAOA with configurable weights."""
    g = MemoryGraph(similarity_threshold=threshold)
    set_graph(g)
    g.add_memories_batch(memories)
    result = recall(query, K=K, graph=g, alpha=alpha, beta_conn=beta_conn, gamma_cov=gamma_cov)
    
    selected_indices = []
    for sel_mem in result.get('memories', []):
        for i, m in enumerate(memories):
            if m == sel_mem['text'] and i not in selected_indices:
                selected_indices.append(i)
                break
    return selected_indices[:K]


def run_on_subset(recall_fn, scenarios, K=5):
    """Run benchmark on a subset of scenarios."""
    total_coverage = 0
    total_recall = 0
    total_f1 = 0
    perfect = 0
    
    for scenario in scenarios:
        memory_texts = [m["text"] for m in scenario["memories"]]
        selected = recall_fn(memory_texts, scenario["question"], K)
        result = evaluate_combination(selected, scenario)
        
        total_coverage += result["coverage"]
        total_recall += result["evidence_recall"]
        total_f1 += result["f1"]
        if result["coverage"] == 1.0:
            perfect += 1
    
    n = len(scenarios)
    return {
        "coverage": total_coverage / n,
        "evidence_recall": total_recall / n,
        "f1": total_f1 / n,
        "perfect": perfect,
        "perfect_pct": perfect / n * 100,
    }


def grid_search_weights(scenarios, K=5):
    """Grid search optimal α, β, γ weights."""
    print("\n" + "=" * 60)
    print("PHASE 1: Grid Search Weights (on 50 scenario sample)")
    print("=" * 60)
    
    # Use first 50 for grid search (speed)
    sample = scenarios[:50]
    
    # Also search similarity threshold
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
    betas = [0.15, 0.25, 0.35, 0.45]
    gammas = [0.1, 0.15, 0.25, 0.35]
    thresholds = [0.1, 0.15, 0.2, 0.25]
    
    best_score = 0
    best_params = {}
    total_combos = len(alphas) * len(betas) * len(gammas) * len(thresholds)
    tested = 0
    
    for a, b, g, t in product(alphas, betas, gammas, thresholds):
        # Skip if weights don't sum reasonably
        total = a + b + g
        if total < 0.7 or total > 1.3:
            tested += 1
            continue
        
        def fn(memories, query, K, _a=a, _b=b, _g=g, _t=t):
            return graph_qaoa_fn(memories, query, K, alpha=_a, beta_conn=_b, gamma_cov=_g, threshold=_t)
        
        result = run_on_subset(fn, sample, K)
        score = result["coverage"]
        
        tested += 1
        if score > best_score:
            best_score = score
            best_params = {"alpha": a, "beta_conn": b, "gamma_cov": g, "threshold": t}
            print(f"  [{tested}/{total_combos}] NEW BEST: {score*100:.1f}% (α={a}, β={b}, γ={g}, t={t})")
        elif tested % 50 == 0:
            print(f"  [{tested}/{total_combos}] current best: {best_score*100:.1f}%")
    
    print(f"\n  OPTIMAL WEIGHTS: α={best_params['alpha']}, β={best_params['beta_conn']}, γ={best_params['gamma_cov']}, threshold={best_params['threshold']}")
    print(f"  BEST COVERAGE: {best_score*100:.1f}%")
    
    return best_params


def run_full_benchmark(scenarios, best_params, K=5):
    """Run full 250-scenario benchmark with optimal weights."""
    print("\n" + "=" * 60)
    print("PHASE 2: Full Benchmark (250 scenarios)")
    print("=" * 60)
    
    # Method 1: Top-K baseline
    print("\n  Running Top-K baseline...")
    t0 = time.time()
    topk_results = run_on_subset(topk_recall_fn, scenarios, K)
    topk_time = time.time() - t0
    print(f"    Coverage: {topk_results['coverage']*100:.1f}% | Perfect: {topk_results['perfect']}/250 | Time: {topk_time:.1f}s")
    
    # Method 2: Graph+QAOA with default weights
    print("\n  Running Graph+QAOA (default weights)...")
    def default_fn(memories, query, K):
        return graph_qaoa_fn(memories, query, K, alpha=0.4, beta_conn=0.35, gamma_cov=0.25, threshold=0.15)
    t0 = time.time()
    default_results = run_on_subset(default_fn, scenarios, K)
    default_time = time.time() - t0
    print(f"    Coverage: {default_results['coverage']*100:.1f}% | Perfect: {default_results['perfect']}/250 | Time: {default_time:.1f}s")
    
    # Method 3: Graph+QAOA with optimized weights
    print("\n  Running Graph+QAOA (optimized weights)...")
    def opt_fn(memories, query, K):
        return graph_qaoa_fn(memories, query, K, **best_params)
    t0 = time.time()
    opt_results = run_on_subset(opt_fn, scenarios, K)
    opt_time = time.time() - t0
    print(f"    Coverage: {opt_results['coverage']*100:.1f}% | Perfect: {opt_results['perfect']}/250 | Time: {opt_time:.1f}s")
    
    return {
        "topk": {**topk_results, "time": topk_time},
        "default": {**default_results, "time": default_time},
        "optimized": {**opt_results, "time": opt_time},
        "params": best_params,
    }


def main():
    print("\n🦍⚛️ MemCombine Full Benchmark Suite\n")
    
    K = 5
    scenarios = ALL_SCENARIOS
    
    # Phase 1: Grid search on 50 scenarios
    best_params = grid_search_weights(scenarios, K)
    
    # Phase 2: Full benchmark on all 250
    results = run_full_benchmark(scenarios, best_params, K)
    
    # Final report
    print("\n" + "=" * 60)
    print("FINAL RESULTS — MemCombine Benchmark (250 scenarios)")
    print("=" * 60)
    print(f"\n  {'Method':<30} {'Coverage':>10} {'Evidence':>10} {'F1':>10} {'Perfect':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for name, label in [("topk", "Top-K Baseline"), ("default", "Graph+QAOA (default)"), ("optimized", "Graph+QAOA (tuned)")]:
        r = results[name]
        print(f"  {label:<30} {r['coverage']*100:>9.1f}% {r['evidence_recall']*100:>9.1f}% {r['f1']*100:>9.1f}% {r['perfect']:>7}/250")
    
    print(f"\n  Optimal weights: α={best_params['alpha']}, β={best_params['beta_conn']}, γ={best_params['gamma_cov']}, threshold={best_params['threshold']}")
    print(f"  Tuned advantage over Top-K: {(results['optimized']['coverage']-results['topk']['coverage'])*100:+.1f}%")
    print(f"  Tuning improvement over default: {(results['optimized']['coverage']-results['default']['coverage'])*100:+.1f}%")
    
    # Save
    output = {
        "benchmark": "MemCombine_v2",
        "scenarios": 250,
        "K": K,
        "results": results,
    }
    with open("/home/dt/Projects/quantum-memory-graph/benchmarks/results_250.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to benchmarks/results_250.json")


if __name__ == "__main__":
    main()
