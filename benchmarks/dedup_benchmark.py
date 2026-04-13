#!/usr/bin/env python3
"""
Dedup Benchmark — Insert duplicates, measure recall before/after dedup.

Strategy: 
  1. Load LongMemEval scenarios
  2. Store all memories (normal)
  3. Benchmark recall → baseline scores
  4. Insert duplicates (2x-3x of existing memories)  
  5. Benchmark recall → polluted scores
  6. Run dedup
  7. Benchmark recall → cleaned scores
  8. Compare all three

DK 🦍
"""

import json
import time
import os
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_memory_graph.graph import MemoryGraph
from quantum_memory_graph.pipeline import recall, set_graph, get_stm
from quantum_memory_graph.dedup import MemoryDeduplicator


def load_scenarios(path="benchmarks/memcombine_250.json"):
    with open(path) as f:
        return json.load(f)


def benchmark_recall(graph, scenarios, K=5, label=""):
    """Run recall benchmark and return hit stats."""
    total = 0
    hits_at_5 = 0
    hits_at_10 = 0
    ndcg_sum = 0
    
    set_graph(graph)
    
    for scenario in scenarios:
        query = scenario.get("query", scenario.get("question", ""))
        expected = scenario.get("expected_memories", scenario.get("relevant", []))
        
        if not query or not expected:
            continue
        
        total += 1
        result = recall(query=query, K=10, hops=2, top_seeds=5,
                       max_candidates=14, use_recency=False)
        
        retrieved_texts = [m["text"] for m in result.get("memories", [])]
        
        # R@5
        top5 = retrieved_texts[:5]
        for exp in expected:
            if any(exp.lower() in r.lower() or r.lower() in exp.lower() for r in top5):
                hits_at_5 += 1
                break
        
        # R@10
        top10 = retrieved_texts[:10]
        for exp in expected:
            if any(exp.lower() in r.lower() or r.lower() in exp.lower() for r in top10):
                hits_at_10 += 1
                break
    
    r5 = (hits_at_5 / total * 100) if total > 0 else 0
    r10 = (hits_at_10 / total * 100) if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Queries: {total}")
    print(f"  R@5:  {r5:.1f}%  ({hits_at_5}/{total})")
    print(f"  R@10: {r10:.1f}%  ({hits_at_10}/{total})")
    print(f"  Graph: {graph.graph.number_of_nodes()} nodes, {graph.graph.number_of_edges()} edges")
    
    return {"r5": r5, "r10": r10, "total": total, "nodes": graph.graph.number_of_nodes(),
            "edges": graph.graph.number_of_edges(), "label": label}


def run_dedup_benchmark():
    print("🦍 Dedup Benchmark — Quantum Memory Graph v1.0.0")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load scenarios
    if os.path.exists("benchmarks/memcombine_250.json"):
        scenarios = load_scenarios("benchmarks/memcombine_250.json")
    else:
        print("   ⚠️ No scenario file found. Using generated test data.")
        scenarios = []
    
    if not scenarios:
        print("   No scenarios to benchmark. Generate with benchmarks/generate_scenarios.py")
        return
    
    # Extract all memory texts from scenarios
    all_texts = set()
    for s in scenarios:
        for m in s.get("memories", s.get("context", [])):
            if isinstance(m, str):
                all_texts.add(m)
            elif isinstance(m, dict):
                all_texts.add(m.get("text", m.get("memory", "")))
    
    all_texts = [t for t in all_texts if t]
    print(f"   Unique memories: {len(all_texts)}")
    
    # Phase 1: Baseline (clean data)
    print("\n🏁 Phase 1: Loading clean data...")
    graph = MemoryGraph(similarity_threshold=0.3)
    graph.add_memories_batch(all_texts)
    baseline = benchmark_recall(graph, scenarios, label="BASELINE (clean data)")
    
    # Phase 2: Insert duplicates
    print("\n💀 Phase 2: Inserting duplicates...")
    import random
    dupes_to_add = []
    for text in all_texts:
        # Add 1-2 near-duplicates per memory
        for _ in range(random.randint(1, 2)):
            # Slight variations
            variant = random.choice([
                text,  # Exact duplicate
                text + ".",  # Trailing period
                text.strip() + " ",  # Trailing space (different hash)
                "Note: " + text,  # Prefix
            ])
            dupes_to_add.append(variant)
    
    graph.add_memories_batch(dupes_to_add)
    polluted = benchmark_recall(graph, scenarios, label="POLLUTED (with duplicates)")
    
    # Phase 3: Dedup
    print("\n🧹 Phase 3: Running deduplication...")
    deduper = MemoryDeduplicator(threshold=0.95)
    
    # First dry run
    dry_stats = deduper.merge_duplicates(graph, dry_run=True)
    print(f"   Dry run: {dry_stats['groups_found']} duplicate groups, "
          f"{dry_stats['duplicates_removed']} would be removed")
    
    # Real merge
    merge_stats = deduper.merge_duplicates(graph)
    print(f"   Merged: {merge_stats['duplicates_removed']} duplicates removed")
    print(f"   Entities merged: {merge_stats['entities_merged']}")
    print(f"   Before: {merge_stats['memories_before']} → After: {merge_stats['memories_after']}")
    
    cleaned = benchmark_recall(graph, scenarios, label="CLEANED (after dedup)")
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  {'Phase':<25} {'R@5':>8} {'R@10':>8} {'Nodes':>8} {'Edges':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in [baseline, polluted, cleaned]:
        print(f"  {r['label'][:25]:<25} {r['r5']:>7.1f}% {r['r10']:>7.1f}% {r['nodes']:>8} {r['edges']:>8}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": baseline,
        "polluted": polluted,
        "cleaned": cleaned,
        "dedup_stats": merge_stats,
    }
    
    with open("benchmarks/dedup_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n   Results saved to benchmarks/dedup_results.json")
    print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run_dedup_benchmark()
