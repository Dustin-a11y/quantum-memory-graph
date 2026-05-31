"""
Pipeline — The unified store/recall interface.

This is what users interact with:
  store(text) → builds graph
  recall(query, K) → graph neighborhood → QAOA subgraph → optimal memories

Copyright 2026 Coinkong (Chef's Attraction). MIT License.
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from .graph import MemoryGraph
from .subgraph_optimizer import optimize_subgraph as _std_optimize_subgraph
from .recency import ShortTermMemory


# Module-level default graph instance
_default_graph: Optional[MemoryGraph] = None
_default_stm: Optional[ShortTermMemory] = None


def get_graph(similarity_threshold: float = 0.3) -> MemoryGraph:
    """Get or create the default memory graph."""
    global _default_graph
    if _default_graph is None:
        _default_graph = MemoryGraph(similarity_threshold=similarity_threshold)
    return _default_graph


def set_graph(graph: MemoryGraph):
    """Set a custom graph as the default."""
    global _default_graph
    _default_graph = graph


def get_stm(**kwargs) -> ShortTermMemory:
    """Get or create the default short-term memory layer."""
    global _default_stm
    if _default_stm is None:
        _default_stm = ShortTermMemory(**kwargs)
    return _default_stm


def set_stm(stm: ShortTermMemory):
    """Set a custom short-term memory layer."""
    global _default_stm
    _default_stm = stm


def store(
    text: str,
    entities: List[str] = None,
    timestamp: datetime = None,
    source: str = "",
    metadata: Dict = None,
    graph: MemoryGraph = None,
) -> Dict:
    """
    Store a memory in the knowledge graph.
    
    The memory is embedded, entities are extracted, and it's connected
    to all related existing memories via weighted edges.
    
    Args:
        text: Memory content
        entities: Known entities (auto-extracted if not provided)
        timestamp: When this happened
        source: Source identifier (conversation ID, document name, etc.)
        metadata: Additional metadata
        graph: Optional custom graph (uses default if not provided)
    
    Returns:
        Dict with memory ID and graph stats
    """
    g = graph or get_graph()
    memory = g.add_memory(
        text=text, entities=entities, timestamp=timestamp,
        source=source, metadata=metadata,
    )
    
    # Update short-term memory
    stm = get_stm()
    stm.on_store(memory.id, text, timestamp)
    
    # Count edges for this memory
    edges = list(g.graph.edges(memory.id, data=True))
    connections = [
        {
            "to": e[1],
            "weight": round(e[2].get("weight", 0), 3),
            "types": e[2].get("types", []),
        }
        for e in sorted(edges, key=lambda x: x[2].get("weight", 0), reverse=True)[:5]
    ]
    
    return {
        "ok": True,
        "memory_id": memory.id,
        "entities": memory.entities,
        "connections": len(edges),
        "top_connections": connections,
        "graph_stats": g.stats(),
    }


def store_batch(
    texts: List[str],
    entities_list: List[List[str]] = None,
    timestamps: List[datetime] = None,
    sources: List[str] = None,
    graph: MemoryGraph = None,
) -> Dict:
    """Batch store memories."""
    g = graph or get_graph()
    memories = g.add_memories_batch(
        texts=texts, entities_list=entities_list,
        timestamps=timestamps, sources=sources,
    )
    return {
        "ok": True,
        "stored": len(memories),
        "memory_ids": [m.id for m in memories],
        "graph_stats": g.stats(),
    }


def recall(
    query: str,
    K: int = 5,
    hops: int = 2,
    top_seeds: int = 5,
    alpha: float = 0.4,
    beta_conn: float = 0.35,
    gamma_cov: float = 0.25,
    graph: MemoryGraph = None,
    max_candidates: int = 14,
    use_recency: bool = True,
    stm: ShortTermMemory = None,
    method: str = "qaoa",
) -> Dict:
    """
    Recall optimal memories for a query.
    
    Pipeline:
      1. Graph neighborhood search (embedding + multi-hop traversal)
      2. Build candidate subgraph (adjacency + relevance)
      3. QAOA optimal subgraph selection
    
    This finds memories that are not just individually relevant,
    but collectively form the best context for the query.
    
    Args:
        query: What to recall
        K: Number of memories to return
        hops: Graph traversal depth
        top_seeds: Initial seed nodes from embedding search
        alpha: Relevance weight in cost function
        beta_conn: Connectivity weight
        gamma_cov: Coverage/diversity weight
        graph: Optional custom graph
        max_candidates: Max nodes for QAOA (caps qubit count)
    
    Returns:
        Dict with selected memories, scores, method details
    """
    g = graph or get_graph()
    
    if not g.memories:
        return {"ok": True, "memories": [], "method": "empty"}
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    def _tier(mem):
        """Determine if a memory is warm (today) or cold (older)."""
        if mem.timestamp and today_str in str(mem.timestamp):
            return "warm"
        return "cold"
    
    # Phase 1: Graph neighborhood search
    neighborhood = g.get_neighborhood(
        query=query, hops=hops, top_seeds=top_seeds
    )
    
    if not neighborhood:
        return {"ok": True, "memories": [], "method": "no_matches"}
    
    # Apply short-term memory boosts (recency, conversation context)
    if use_recency:
        _stm = stm or get_stm()
        neighborhood = _stm.apply(neighborhood, g.memories)
    
    # Sort by combined score (embedding + graph traversal + recency)
    sorted_candidates = sorted(
        neighborhood.items(), key=lambda x: x[1], reverse=True
    )[:max_candidates]
    
    candidate_ids = [cid for cid, _ in sorted_candidates]
    candidate_scores = np.array([score for _, score in sorted_candidates])
    
    if len(candidate_ids) <= K:
        memories = [
            {
                "text": g.memories[cid].text,
                "entities": g.memories[cid].entities,
                "relevance": float(candidate_scores[i]),
                "source": g.memories[cid].source,
                "tier": _tier(g.memories[cid]),
            }
            for i, cid in enumerate(candidate_ids)
        ]
        return {
            "ok": True,
            "memories": memories,
            "method": "all_candidates",
            "candidates": len(candidate_ids),
            "tier_counts": {
                "warm": sum(1 for m in memories if m["tier"] == "warm"),
                "cold": sum(1 for m in memories if m["tier"] == "cold"),
            },
        }
    
    # Phase 2: Synergy rerank or QAOA subgraph
    if method == "synergy":
        try:
            from .synergy_reranker import select as synergy_select
            # Build texts for candidates
            cand_texts = [g.memories[cid].text for cid in candidate_ids]
            selected_synergy = synergy_select(candidate_scores, cand_texts, query, K)
            
            selected_memories = []
            for idx in selected_synergy:
                cid = candidate_ids[idx]
                mem = g.memories[cid]
                selected_memories.append({
                    "text": mem.text,
                    "entities": mem.entities,
                    "relevance": float(candidate_scores[idx]),
                    "source": mem.source,
                    "tier": _tier(mem),
                    "connections": [],
                })
            return {
                "ok": True,
                "memories": selected_memories,
                "method": "synergy",
                "candidates": len(candidate_ids),
                "K": K,
            }
        except Exception as e:
            print(f"WARNING: Synergy selection failed ({e}), falling back to QAOA")
    
    # Phase 3: QAOA subgraph optimization
    subgraph = g.get_subgraph_data(candidate_ids)
    adjacency = subgraph["adjacency"]
    # Use PCE for larger candidate sets (14+), standard QAOA for smaller
    try:
        if len(candidate_ids) > 14:
            from .pce_optimizer import optimize_subgraph_pce
            result = optimize_subgraph_pce(
                relevance_scores=candidate_scores,
                adjacency=adjacency,
                K=K,
                alpha=alpha,
                beta_conn=beta_conn,
                gamma_cov=gamma_cov,
            )
        else:
            result = _std_optimize_subgraph(
                relevance_scores=candidate_scores,
                adjacency=adjacency,
                K=K,
                alpha=alpha,
                beta_conn=beta_conn,
                gamma_cov=gamma_cov,
            )
    except Exception as e:
        # Ultimate fallback: use greedy if QAOA fails despite internal try/except
        print(f"WARNING: Subgraph optimization failed ({e}), using greedy fallback")
        from .subgraph_optimizer import _greedy_subgraph
        greedy_sel, greedy_score = _greedy_subgraph(
            candidate_scores, adjacency, K, alpha, beta_conn, gamma_cov
        )
        result = {
            "selection": greedy_sel,
            "score": greedy_score,
            "greedy": {"selection": greedy_sel, "score": greedy_score},
            "optimal": {"selection": greedy_sel, "score": greedy_score},
            "method": "greedy_safety_net",
        }
    
    selected_idxs = result["selection"]
    selected_memories = []
    
    # Compute comparison pcts safely
    qaoa_score_val = result.get("qaoa", {}).get("score", result["score"])
    greedy_score_val = result.get("greedy", {}).get("score", result["score"])
    optimal_score_val = result.get("optimal", {}).get("score", result["score"])
    qaoa_vs_greedy = (qaoa_score_val / greedy_score_val * 100) if greedy_score_val > 0 else 100
    qaoa_vs_optimal = (qaoa_score_val / optimal_score_val * 100) if optimal_score_val > 0 else 100
    
    for idx in selected_idxs:
        cid = candidate_ids[idx]
        mem = g.memories[cid]
        
        # Find connections to other selected memories
        connections = []
        for other_idx in selected_idxs:
            if other_idx != idx:
                other_cid = candidate_ids[other_idx]
                if g.graph.has_edge(cid, other_cid):
                    edge_data = g.graph[cid][other_cid]
                    connections.append({
                        "to_text": g.memories[other_cid].text[:80],
                        "weight": round(edge_data.get("weight", 0), 3),
                        "types": edge_data.get("types", []),
                    })
        
        selected_memories.append({
            "text": mem.text,
            "entities": mem.entities,
            "relevance": float(candidate_scores[idx]),
            "source": mem.source,
            "tier": _tier(mem),
            "connections": connections,
        })
    
    return {
        "ok": True,
        "memories": selected_memories,
        "method": "qaoa_subgraph",
        "candidates": len(candidate_ids),
        "K": K,
        "qaoa_score": result["score"],
        "greedy_score": result["greedy"]["score"],
        "optimal_score": result["optimal"]["score"],
        "qaoa_vs_optimal_pct": round(float(qaoa_vs_optimal), 2),
        "qaoa_vs_greedy_pct": round(float(qaoa_vs_greedy), 2),
        "tier_counts": {
            "warm": sum(1 for m in selected_memories if m["tier"] == "warm"),
            "cold": sum(1 for m in selected_memories if m["tier"] == "cold"),
        },
        "graph_stats": g.stats(),
    }
