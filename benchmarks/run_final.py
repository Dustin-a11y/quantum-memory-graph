"""
Final MemCombine benchmark — skip grid search, use known optimal weights.
3-way comparison on all 250 scenarios.

DK 🦍
"""

import json, sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_memory_graph.graph import MemoryGraph
from quantum_memory_graph.pipeline import recall, store_batch, set_graph
from benchmarks.memcombine import evaluate_combination

print("Loading scenarios...", flush=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "memcombine_250.json")) as f:
    ALL = json.load(f)
print(f"Loaded {len(ALL)} scenarios", flush=True)

print("Loading model...", flush=True)
from sentence_transformers import SentenceTransformer
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready.\n", flush=True)

# Known optimal from grid search
TUNED = {"alpha": 0.3, "beta_conn": 0.25, "gamma_cov": 0.15, "threshold": 0.15}
DEFAULT = {"alpha": 0.4, "beta_conn": 0.35, "gamma_cov": 0.25, "threshold": 0.15}
K = 5

def topk(memories, query, K):
    texts = memories + [query]
    embs = MODEL.encode(texts, normalize_embeddings=True)
    scores = np.dot(embs[:-1], embs[-1])
    return np.argsort(scores)[-K:][::-1].tolist()

def graph_qaoa(memories, query, K, alpha, beta_conn, gamma_cov, threshold):
    g = MemoryGraph(similarity_threshold=threshold)
    g._embedder = MODEL
    g.add_memories_batch(memories)
    set_graph(g)
    result = recall(query, K=K, graph=g, alpha=alpha, beta_conn=beta_conn, gamma_cov=gamma_cov)
    indices = []
    for sel in result.get('memories', []):
        for i, m in enumerate(memories):
            if m == sel['text'] and i not in indices:
                indices.append(i)
                break
    return indices[:K]

def run_method(name, fn, scenarios):
    total_cov = 0; total_rec = 0; total_f1 = 0; perfect = 0
    for i, s in enumerate(scenarios):
        texts = [m["text"] for m in s["memories"]]
        selected = fn(texts, s["question"], K)
        r = evaluate_combination(selected, s)
        total_cov += r["coverage"]; total_rec += r["evidence_recall"]
        total_f1 += r["f1"]; perfect += (1 if r["coverage"] == 1.0 else 0)
        if (i+1) % 50 == 0:
            n = i+1
            print(f"  [{name}] {n}/250 — cov={total_cov/n*100:.1f}% perfect={perfect}/{n}", flush=True)
    n = len(scenarios)
    return {"coverage": total_cov/n, "evidence_recall": total_rec/n, "f1": total_f1/n, "perfect": perfect, "perfect_pct": perfect/n*100}

print("=" * 60, flush=True)
print("MemCombine — 250 Scenarios, K=5", flush=True)
print("=" * 60, flush=True)

print("\n1. Top-K Baseline...", flush=True)
t0 = time.time()
r_topk = run_method("TopK", topk, ALL)
print(f"   Done in {time.time()-t0:.0f}s\n", flush=True)

print("2. Graph+QAOA (default)...", flush=True)
t0 = time.time()
r_default = run_method("Default", lambda m,q,k: graph_qaoa(m,q,k,**DEFAULT), ALL)
print(f"   Done in {time.time()-t0:.0f}s\n", flush=True)

print("3. Graph+QAOA (tuned)...", flush=True)
t0 = time.time()
r_tuned = run_method("Tuned", lambda m,q,k: graph_qaoa(m,q,k,**TUNED), ALL)
print(f"   Done in {time.time()-t0:.0f}s\n", flush=True)

print("=" * 60, flush=True)
print("FINAL RESULTS", flush=True)
print("=" * 60, flush=True)
print(f"\n  {'Method':<30} {'Coverage':>10} {'Evidence':>10} {'F1':>10} {'Perfect':>10}", flush=True)
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", flush=True)
for name, r in [("Top-K Baseline", r_topk), ("Graph+QAOA (default)", r_default), ("Graph+QAOA (tuned)", r_tuned)]:
    print(f"  {name:<30} {r['coverage']*100:>9.1f}% {r['evidence_recall']*100:>9.1f}% {r['f1']*100:>9.1f}% {r['perfect']:>7}/250", flush=True)

print(f"\n  Tuned weights: α={TUNED['alpha']}, β={TUNED['beta_conn']}, γ={TUNED['gamma_cov']}, threshold={TUNED['threshold']}", flush=True)
print(f"  Tuned vs Top-K: {(r_tuned['coverage']-r_topk['coverage'])*100:+.1f}%", flush=True)
print(f"  Tuned vs Default: {(r_tuned['coverage']-r_default['coverage'])*100:+.1f}%", flush=True)

output = {"benchmark": "MemCombine", "scenarios": 250, "K": K,
          "topk": r_topk, "default": r_default, "tuned": r_tuned, "params": TUNED}
with open("/home/dt/Projects/quantum-memory-graph/benchmarks/results_250.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to results_250.json", flush=True)
