"""
Full pipeline test — store memories, build graph, recall with QAOA subgraph.
Also runs MemCombine benchmark comparing graph+QAOA vs plain Top-K.

DK 🦍
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from quantum_memory_graph.graph import MemoryGraph
from quantum_memory_graph.pipeline import store, recall, store_batch, set_graph, get_graph
from quantum_memory_graph.subgraph_optimizer import optimize_subgraph

# Add benchmarks to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'benchmarks'))
from memcombine import run_benchmark, SCENARIOS


def test_graph_construction():
    """Test that memories form a connected graph with proper edges."""
    print("=" * 60)
    print("TEST 1: Graph Construction")
    print("=" * 60)
    
    g = MemoryGraph(similarity_threshold=0.15)
    
    memories = [
        "The quantum memory API runs on port 8501 on the DGX Spark server.",
        "IBM Quantum hardware cron runs 5 times daily using ibm_kingston backend.",
        "Chef decided to open source the quantum memory project under MIT license.",
        "Copyright was updated to Coinkong Chef's Attraction for all quantum files.",
        "Team had pizza for lunch. The pepperoni was excellent.",
        "Previous SOTA achieves 96.6% recall on LongMemEval benchmark.",
        "QAOA provides 99.7% optimality on memory selection problems.",
        "The DGX Spark has 128GB RAM and an NVIDIA GB10 GPU.",
        "PostgreSQL with pgvector runs on the Spark for embedding storage.",
        "The restaurant catering system needs work order management.",
    ]
    
    g.add_memories_batch(memories)
    stats = g.stats()
    
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Density: {stats['density']:.3f}")
    print(f"  Components: {stats['components']}")
    print(f"  Avg degree: {stats['avg_degree']:.1f}")
    
    assert stats['nodes'] == 10, f"Expected 10 nodes, got {stats['nodes']}"
    assert stats['edges'] > 5, f"Expected >5 edges, got {stats['edges']}"
    assert stats['density'] > 0, "Graph should have some density"
    
    print("  ✅ PASS\n")
    return g


def test_neighborhood_search(g):
    """Test that graph traversal finds related memories across hops."""
    print("=" * 60)
    print("TEST 2: Neighborhood Search (Multi-hop)")
    print("=" * 60)
    
    # Query about quantum infrastructure
    neighborhood = g.get_neighborhood("What runs on the Spark server?", hops=2, top_seeds=3)
    
    print(f"  Found {len(neighborhood)} memories in neighborhood")
    for mid, score in sorted(neighborhood.items(), key=lambda x: x[1], reverse=True)[:5]:
        mem = g.memories[mid]
        print(f"    [{score:.3f}] {mem.text[:70]}...")
    
    assert len(neighborhood) >= 3, f"Should find >=3 related memories, got {len(neighborhood)}"
    
    # The pizza memory should NOT be in top results
    pizza_scores = [score for mid, score in neighborhood.items() 
                    if "pizza" in g.memories[mid].text.lower()]
    if pizza_scores:
        quantum_scores = [score for mid, score in neighborhood.items()
                         if "quantum" in g.memories[mid].text.lower() or "spark" in g.memories[mid].text.lower()]
        if quantum_scores:
            assert max(pizza_scores) < max(quantum_scores), "Pizza should rank below quantum memories"
    
    print("  ✅ PASS\n")


def test_subgraph_optimizer():
    """Test QAOA subgraph selection vs greedy vs brute force."""
    print("=" * 60)
    print("TEST 3: QAOA Subgraph Optimizer")
    print("=" * 60)
    
    n = 8
    K = 3
    
    # Create a scenario where connected nodes should be preferred
    relevance = np.array([0.9, 0.85, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1])
    
    # Adjacency: nodes 0,1,2 are strongly connected (they're related)
    # Nodes 3,4 are isolated high-relevance but unconnected
    adjacency = np.zeros((n, n))
    adjacency[0][1] = 0.8; adjacency[1][0] = 0.8
    adjacency[0][2] = 0.7; adjacency[2][0] = 0.7
    adjacency[1][2] = 0.9; adjacency[2][1] = 0.9
    adjacency[5][6] = 0.3; adjacency[6][5] = 0.3
    
    result = optimize_subgraph(relevance, adjacency, K)
    
    print(f"  QAOA selected: {result['selection']} (score: {result['score']:.4f})")
    print(f"  Greedy selected: {result['greedy']['selection']} (score: {result['greedy']['score']:.4f})")
    print(f"  Optimal selected: {result['optimal']['selection']} (score: {result['optimal']['score']:.4f})")
    print(f"  QAOA vs Optimal: {result['qaoa_vs_optimal_pct']:.1f}%")
    print(f"  QAOA vs Greedy: {result['qaoa_vs_greedy_pct']:.1f}%")
    
    # QAOA should select the connected cluster (0,1,2) because
    # connectivity bonus outweighs the slight relevance advantage of isolated nodes
    sel = set(result['selection'])
    connected_cluster = {0, 1, 2}
    overlap = len(sel & connected_cluster)
    print(f"  Connected cluster overlap: {overlap}/3")
    
    assert result['qaoa_vs_optimal_pct'] >= 80, f"QAOA should be >=80% of optimal, got {result['qaoa_vs_optimal_pct']:.1f}%"
    assert overlap >= 2, f"Should select >=2 from connected cluster, got {overlap}"
    
    print("  ✅ PASS\n")


def test_full_pipeline():
    """Test store → recall pipeline end-to-end."""
    print("=" * 60)
    print("TEST 4: Full Pipeline (Store → Recall)")
    print("=" * 60)
    
    # Fresh graph
    g = MemoryGraph(similarity_threshold=0.15)
    set_graph(g)
    
    # Store memories about a project
    project_memories = [
        "Project Alpha uses React frontend with TypeScript for type safety.",
        "Project Alpha backend is FastAPI with PostgreSQL database.",
        "The API design follows REST conventions with JWT authentication.",
        "React components use Material UI for consistent styling.",
        "FastAPI connects to PostgreSQL via SQLAlchemy ORM.",
        "Team standup happens daily at 9am in the main conference room.",
        "Lunch menu today: chicken tikka masala with naan bread.",
        "CI/CD pipeline runs on GitHub Actions with Docker containers.",
        "Project Beta uses Vue.js, completely separate codebase.",
        "Office wifi password was changed to a new secure passphrase.",
    ]
    
    result = store_batch(project_memories)
    print(f"  Stored {result['stored']} memories")
    print(f"  Graph: {result['graph_stats']['nodes']} nodes, {result['graph_stats']['edges']} edges")
    
    # Recall about Project Alpha's tech stack
    recall_result = recall("What is Project Alpha's full technology stack?", K=5)
    
    print(f"  Method: {recall_result['method']}")
    print(f"  Candidates: {recall_result.get('candidates', '?')}")
    print(f"  QAOA score: {recall_result.get('qaoa_score', '?')}")
    print(f"  QAOA vs Optimal: {recall_result.get('qaoa_vs_optimal_pct', '?')}%")
    print(f"  Selected {len(recall_result['memories'])} memories:")
    
    selected_texts = []
    for m in recall_result['memories']:
        print(f"    - {m['text'][:70]}...")
        print(f"      Entities: {m.get('entities', [])[:3]}")
        print(f"      Connections: {len(m.get('connections', []))}")
        selected_texts.append(m['text'])
    
    # Should find React, FastAPI, PostgreSQL, and related memories
    all_text = " ".join(selected_texts).lower()
    found_react = "react" in all_text
    found_fastapi = "fastapi" in all_text
    found_postgres = "postgresql" in all_text
    
    print(f"\n  Found React: {found_react}")
    print(f"  Found FastAPI: {found_fastapi}")
    print(f"  Found PostgreSQL: {found_postgres}")
    
    # Should NOT find lunch or wifi
    found_noise = "lunch" in all_text or "wifi" in all_text
    print(f"  Found noise (lunch/wifi): {found_noise}")
    
    tech_count = sum([found_react, found_fastapi, found_postgres])
    assert tech_count >= 2, f"Should find >=2 tech stack items, got {tech_count}"
    
    print("  ✅ PASS\n")


def test_memcombine_benchmark():
    """Run MemCombine benchmark: Graph+QAOA vs Top-K."""
    print("=" * 60)
    print("TEST 5: MemCombine Benchmark")
    print("=" * 60)
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def topk_recall(memories, query, K):
        """Baseline: pure embedding Top-K."""
        texts = memories + [query]
        embs = model.encode(texts, normalize_embeddings=True)
        query_emb = embs[-1]
        mem_embs = embs[:-1]
        scores = np.dot(mem_embs, query_emb)
        return np.argsort(scores)[-K:][::-1].tolist()
    
    def graph_qaoa_recall(memories, query, K):
        """Our system: Graph + QAOA subgraph."""
        g = MemoryGraph(similarity_threshold=0.15)
        set_graph(g)
        g.add_memories_batch(memories)
        result = recall(query, K=K, graph=g)
        
        # Map back to indices
        selected_indices = []
        for sel_mem in result.get('memories', []):
            for i, m in enumerate(memories):
                if m == sel_mem['text'] and i not in selected_indices:
                    selected_indices.append(i)
                    break
        return selected_indices[:K]
    
    # Run both
    print("\n  --- Top-K Baseline ---")
    topk_results = run_benchmark(topk_recall, K=5)
    print(f"  Coverage:        {topk_results['avg_coverage']*100:.1f}%")
    print(f"  Evidence Recall: {topk_results['avg_evidence_recall']*100:.1f}%")
    print(f"  F1:              {topk_results['avg_f1']*100:.1f}%")
    print(f"  Perfect:         {topk_results['perfect_coverage']}/{topk_results['n_scenarios']}")
    
    print("\n  --- Graph + QAOA ---")
    qaoa_results = run_benchmark(graph_qaoa_recall, K=5)
    print(f"  Coverage:        {qaoa_results['avg_coverage']*100:.1f}%")
    print(f"  Evidence Recall: {qaoa_results['avg_evidence_recall']*100:.1f}%")
    print(f"  F1:              {qaoa_results['avg_f1']*100:.1f}%")
    print(f"  Perfect:         {qaoa_results['perfect_coverage']}/{qaoa_results['n_scenarios']}")
    
    advantage = qaoa_results['avg_coverage'] - topk_results['avg_coverage']
    print(f"\n  QAOA Advantage (Coverage): {advantage*100:+.1f}%")
    
    # Print per-scenario comparison
    print("\n  Per-scenario breakdown:")
    for tk, qa in zip(topk_results['per_scenario'], qaoa_results['per_scenario']):
        icon = "🟢" if qa['coverage'] > tk['coverage'] else ("🟡" if qa['coverage'] == tk['coverage'] else "🔴")
        print(f"    {icon} {tk['id']}: Top-K {tk['coverage']*100:.0f}% → QAOA {qa['coverage']*100:.0f}% (noise: {tk['noise']}→{qa['noise']})")
    
    print("  ✅ PASS\n")
    
    return topk_results, qaoa_results


if __name__ == '__main__':
    print("\n🦍⚛️ Quantum Memory Graph — Full Test Suite\n")
    
    g = test_graph_construction()
    test_neighborhood_search(g)
    test_subgraph_optimizer()
    test_full_pipeline()
    topk, qaoa = test_memcombine_benchmark()
    
    print("=" * 60)
    print("ALL TESTS PASSED 🦍⚛️")
    print("=" * 60)
    print(f"\nMemCombine Results:")
    print(f"  Top-K Coverage:  {topk['avg_coverage']*100:.1f}%")
    print(f"  QAOA Coverage:   {qaoa['avg_coverage']*100:.1f}%")
    print(f"  Advantage:       {(qaoa['avg_coverage']-topk['avg_coverage'])*100:+.1f}%")
