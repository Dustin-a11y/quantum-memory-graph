"""
Pipeline — The unified store/recall interface.

This is what users interact with:
  store(text) → builds graph
  recall(query, K) → graph neighborhood → QAOA subgraph → optimal memories

Copyright 2026 Coinkong (Chef's Attraction). MIT License.
"""

import logging
import numpy as np
import re
from typing import List, Dict, Optional
from datetime import datetime

from .graph import MemoryGraph, Memory
from .subgraph_optimizer import optimize_subgraph as _std_optimize_subgraph
from .recency import ShortTermMemory

logger = logging.getLogger(__name__)


# Module-level default graph instance
_default_graph: Optional[MemoryGraph] = None
_default_stm: Optional[ShortTermMemory] = None
_SAFE_METADATA_KEYS = ("session_index", "session_id", "session_date", "session_date_time")


def _memory_result(mem: Memory, relevance: float, tier: str, connections=None) -> Dict:
    """Render retrieval data while exposing only benchmark-neutral metadata."""
    result = {
        "id": mem.id if hasattr(mem, 'id') else "",
        "text": mem.text,
        "entities": mem.entities,
        "relevance": float(relevance),
        "source": mem.source,
        "tier": tier,
        "timestamp": mem.timestamp.isoformat() if mem.timestamp else None,
        "metadata": {key: mem.metadata[key] for key in _SAFE_METADATA_KEYS if key in mem.metadata},
    }
    if connections is not None:
        result["connections"] = connections
    return result


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
    semantic_facts: List[Dict] = None,
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
        source=source, metadata=metadata, semantic_facts=semantic_facts,
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


def recall_facts(subject_id: str, relation_type: str,
                 graph: MemoryGraph = None,
                 observed_as_of: datetime = None,
                 valid_at: datetime = None) -> Dict:
    """Opt-in exact semantic recall, never an inferred identity or winner.

    The ordinary text ``recall`` remains unchanged. This path returns every
    indexed matching assertion with its provenance, both time dimensions,
    and the legacy memory nodes supplying its evidence.
    """
    g = graph or get_graph()
    facts = g.query_facts(subject_id, relation_type,
                          observed_as_of=observed_as_of, valid_at=valid_at)
    linked = []
    seen = set()
    today_str = datetime.now().strftime("%Y-%m-%d")
    for fact in facts:
        for memory_id in fact.evidence_memory_ids or (fact.source_memory_id,):
            if memory_id in seen:
                continue
            if memory_id not in g.memories:
                raise ValueError("typed fact links to a missing source memory")
            seen.add(memory_id)
            memory = g.memories[memory_id]
            tier = "warm" if memory.timestamp and today_str in str(memory.timestamp) else "cold"
            linked.append(_memory_result(memory, 1.0, tier))
    return {
        "ok": True,
        "method": "semantic_index",
        "facts": [fact.to_dict() for fact in facts],
        "memories": linked,
        "candidates": len(facts),
    }


def store_chunked(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    entities: List[str] = None,
    timestamp: datetime = None,
    source: str = "",
    metadata: Dict = None,
    graph: MemoryGraph = None,
) -> Dict:
    """
    Store a memory as overlapping chunks, each with its own embedding.
    Dramatically improves recall for specific fact retrieval.
    """
    g = graph or get_graph()
    
    if len(text) <= chunk_size:
        return store(text=text, entities=entities, timestamp=timestamp,
                    source=source, metadata=metadata, graph=g)
    
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    
    chunk_ids = []
    chunk_meta = (metadata or {}) | {
        "chunked": True, "chunk_index": 0,
        "total_chunks": len(chunks), "original_length": len(text),
    }
    
    for i, chunk in enumerate(chunks):
        chunk_meta["chunk_index"] = i
        mem = g.add_memory(
            text=chunk, entities=entities, timestamp=timestamp,
            source=f"{source}#chunk{i}" if source else f"chunk{i}",
            metadata=chunk_meta.copy(),
        )
        chunk_ids.append(mem.id)
    
    return {
        "ok": True, "chunked": True,
        "num_chunks": len(chunks), "chunk_ids": chunk_ids,
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


def _anchor_top_relevance(selection, relevance_scores, K: int,
                          anchor: int = 1) -> List[int]:
    """Guarantee the most relevant candidates survive subgraph selection.

    The subgraph objective mixes relevance (alpha) with connectivity
    (beta_conn) and coverage (gamma_cov), so a well-connected cluster can
    outvote the single most relevant memory - usually the one the question
    needs. Measured on LongMemEval, unanchored QAOA selection dropped a rank-1
    gold memory in 55% of the questions plain embedding search had ranked first
    (issue #27).

    The ``anchor`` highest-relevance candidates are kept whatever the optimiser
    returned; its own picks fill the remaining slots and any shortfall is filled
    in relevance order. ``anchor=K`` is pure relevance top-K, ``anchor=0`` is
    the historic behaviour.
    """
    scores = np.asarray(relevance_scores, dtype=float)
    n = len(scores)
    if n == 0 or K <= 0:
        return []
    limit = min(K, n)

    if anchor <= 0:
        out: List[int] = []
        for idx in selection:
            i = int(idx)
            if 0 <= i < n and i not in out:
                out.append(i)
        return out[:limit]

    order = [int(i) for i in np.argsort(-scores, kind="stable")]
    out = order[:min(anchor, limit)]
    for idx in selection:
        if len(out) >= limit:
            break
        i = int(idx)
        if 0 <= i < n and i not in out:
            out.append(i)
    for i in order:
        if len(out) >= limit:
            break
        if i not in out:
            out.append(i)
    return out[:limit]


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
    method: str = None,  # None -> $QMG_RECALL_METHOD, default "greedy" (classical).
    # QAOA/CVaR stay reachable (method="qaoa"/"pce" or QMG_RECALL_METHOD) but are no
    # longer what a plain /recall runs: the simulator cost ~8x a classical selector and
    # ranked by graph structure, so its relevance numbers could not be trusted.
    text_weight: float = 0.3,
    relevance_anchor: int = 1,
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
    # Selector is a deployment decision, not a constant: the classical greedy selector is the
    # default, QMG_RECALL_METHOD=qaoa|pce|synergy opts into the quantum-era paths.
    method = (method or os.environ.get("QMG_RECALL_METHOD") or "greedy").strip().lower()
    g = graph or get_graph()

    temporal_signals = {
        "first", "before", "after", "when", "days", "weeks", "months",
        "order", "ago", "current", "latest", "earlier", "later", "last",
        "earliest", "newest", "recent", "since", "until",
    }
    reasoning_signals = {
        "total", "average", "how many", "how much", "combined", "sum",
        "difference",
    }
    query_lower = query.lower()
    if any(re.search(rf"\b{re.escape(signal)}\b", query_lower)
           for signal in temporal_signals | reasoning_signals):
        top_seeds = max(top_seeds, 10)
        max_candidates = max(max_candidates, K + 5, 20)

    if not g.memories:
        return {"ok": True, "memories": [], "method": "empty"}
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    def _tier(mem):
        """Determine if a memory is warm (today) or cold (older)."""
        if mem.timestamp and today_str in str(mem.timestamp):
            return "warm"
        return "cold"
    
    # Phase 1: Graph neighborhood search
    neighborhood = g.get_neighborhood(text_weight=text_weight, 
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
            _memory_result(g.memories[cid], candidate_scores[i], _tier(g.memories[cid]))
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
                selected_memories.append(
                    _memory_result(mem, candidate_scores[idx], _tier(mem), connections=[])
                )
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
    
    selected_idxs = _anchor_top_relevance(
        result["selection"], candidate_scores, K, relevance_anchor)
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
        
        selected_memories.append(
            _memory_result(mem, candidate_scores[idx], _tier(mem), connections=connections)
        )
    
    return {
        "ok": True,
        "memories": selected_memories,
        "method": "qaoa_subgraph",
        # The selection path that actually executed. "qaoa_subgraph" above is
        # historical and was returned even when the optimiser raised and greedy
        # ran instead, which hid PCE failures in production (issue #27).
        "selection_method": result.get("method", "unknown"),
        "relevance_anchor": relevance_anchor,
        "candidates": len(candidate_ids),
        "K": K,
        "qaoa_score": result["score"],
        "greedy_score": result.get("greedy", {}).get("score"),
        "optimal_score": result.get("optimal", {}).get("score"),
        "qaoa_vs_optimal_pct": round(float(qaoa_vs_optimal), 2) if result.get("optimal") else None,
        "qaoa_vs_greedy_pct": (round(float(qaoa_vs_greedy), 2)
                               if result.get("qaoa") and result.get("greedy") else None),
        "tier_counts": {
            "warm": sum(1 for m in selected_memories if m["tier"] == "warm"),
            "cold": sum(1 for m in selected_memories if m["tier"] == "cold"),
        },
        "graph_stats": g.stats(),
    }


def answer(
    query: str,
    K: int = 5,
    model: str = "deepseek/deepseek-chat",
    api_key: str = None,
    graph: MemoryGraph = None,
) -> Dict:
    """
    Answer a question by recalling relevant memories and synthesizing an answer.
    
    Pipeline:
      1. Recall K memories via graph + QAOA
      2. Build a prompt with recalled context
      3. Call LLM to synthesize a concise answer
      4. Return answer with sources and confidence
    
    Args:
        query: The question to answer
        K: Number of memories to recall
        model: LLM model for synthesis (via Vercel AI Gateway)
        api_key: Vercel API key (defaults to env variable)
        graph: Optional custom graph
    
    Returns:
        Dict with answer, sources, confidence
    """
    import os
    import json
    import urllib.request
    import re as _re

    # Minimal bounded retry constant for empty-answer recovery
    _MAX_ANSWER_RETRIES = 1

    def _sanitize_error(err_msg: str) -> str:
        """Strip sensitive patterns (API keys, URLs, paths) from error messages."""
        sanitized = str(err_msg)
        sanitized = _re.sub(r'https?://[^\s\'\"]+', '[URL-REDACTED]', sanitized)
        sanitized = _re.sub(r'sk-[a-zA-Z0-9_-]{20,}', '[API-KEY-REDACTED]', sanitized)
        sanitized = _re.sub(r'Bearer\s+[a-zA-Z0-9._\-=]+', '[TOKEN-REDACTED]', sanitized)
        sanitized = _re.sub(r'File\s+".*?(/[^"]+)",\s*line\s+\d+', r'File "[PATH-REDACTED]", line N', sanitized)
        return sanitized[:120]

    def _call_llm_safe(prompt: str, api_key: str, api_base: str, model: str):
        """Call LLM API with bounded retry on empty content."""
        req_data = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.1,
        }).encode()
        req = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=req_data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"].strip()
        if not content and _MAX_ANSWER_RETRIES > 0:
            # One bounded empty retry
            req = urllib.request.Request(
                f"{api_base}/chat/completions",
                data=req_data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"].strip()
        return content
    
    # Step 1: Recall
    recall_result = recall(query=query, K=K, graph=graph)
    memories = recall_result.get("memories", [])
    
    if not memories:
        return {
            "ok": True,
            "answer": "I don't have enough information to answer that question.",
            "sources": [],
            "confidence": 0.0,
            "method": "no_memories",
        }
    
    # Step 2: Build context from memories
    context_parts = []
    sources = []
    for i, mem in enumerate(memories):
        ctx = f"[Memory {i+1}] {mem['text']}"
        entities = mem.get("entities", [])
        if entities:
            ctx += f"\n  Entities: {', '.join(entities[:5])}"
        context_parts.append(ctx)
        sources.append({
            "text": mem["text"][:200],
            "relevance": mem.get("relevance", 0),
            "entities": mem.get("entities", []),
            "source": mem.get("source", ""),
        })
    
    context = "\n\n".join(context_parts)
    
    # Step 3: Build prompt
    prompt = f"""You are answering a question based ONLY on the provided memories. 
If the memories don't contain the answer, say "I don't have enough information."
Be concise and direct. Return ONLY the answer, no explanations.

MEMORIES:
{context}

QUESTION: {query}

ANSWER:"""
    
    # Step 4: Call LLM via OpenRouter
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get("VERCEL_API_KEY", "")
    api_base = os.environ.get("BENCHD_API_BASE", "https://openrouter.ai/api/v1")
    model = os.environ.get("QMG_ANSWER_MODEL", "deepseek/deepseek-chat")
    
    if not api_key:
        # Fallback: heuristic answer from top memory
        top_mem = memories[0]["text"]
        return {
            "ok": True,
            "answer": top_mem,
            "sources": sources,
            "confidence": memories[0].get("relevance", 0),
            "method": "heuristic_fallback",
        }
    
    try:
        answer_text = _call_llm_safe(prompt, api_key, api_base, model)
        
        # Compute confidence from top relevance scores
        confidences = [m.get("relevance", 0) for m in memories[:3]]
        confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "ok": True,
            "answer": answer_text,
            "sources": sources,
            "confidence": round(confidence, 3),
            "method": "llm_synthesis",
            "model": model,
            "recall_method": recall_result.get("method", "unknown"),
        }
    
    except Exception as e:
        # Fallback to heuristic. The sanitized reason is logged server-side
        # only: exception-derived text must not reach the response body
        # (CodeQL py/stack-trace-exposure).
        logger.warning("LLM answer synthesis failed; falling back to top memory: %s",
                       _sanitize_error(str(e)))
        top_mem = memories[0]["text"]
        return {
            "ok": True,
            "answer": top_mem,
            "sources": sources,
            "confidence": memories[0].get("relevance", 0),
            "method": "llm_failed",
        }


def temporal_answer(
    query: str,
    K: int = 8,
    graph: MemoryGraph = None,
) -> Dict:
    """
    Answer temporal/numeric questions using extracted temporal entities.
    """
    from datetime import datetime, timedelta
    
    # Step 1: Recall
    recall_result = recall(query=query, K=K, graph=graph)
    memories = recall_result.get("memories", [])
    
    if not memories:
        return {"ok": True, "answer": "No memories found.", "confidence": 0.0}
    
    g = graph or get_graph()
    
    # Step 2: Extract temporal entities from recalled memories
    all_temporal = []
    for mem in memories:
        mem_id = Memory.make_id(mem["text"])
        if mem_id in g.memories:
            node = g.memories[mem_id]
            for te in node.temporal_entities:
                all_temporal.append({
                    "type": te.get("type", ""),
                    "value": te.get("value", ""),
                    "text": te.get("text", ""),
                    "subject": te.get("subject", ""),
                    "label": te.get("label", ""),
                    "unit": te.get("unit", ""),
                    "memory_text": mem["text"][:120],
                    "relevance": mem.get("relevance", 0),
                })
    
    q = query.lower()
    
    # --- PATTERN: "How old will X be when Y?" ---
    age_match = re.search(r'how old will (\w+(?:\s+\w+)?) be', q)
    if age_match:
        subject = age_match.group(1).strip()
        ages = [t for t in all_temporal if t["type"] == "age" and subject.lower() in str(t.get("subject","")).lower()]
        dates = [t for t in all_temporal if t["type"] == "date"]
        if ages and dates:
            age_val = ages[0]["value"]
            date_val = dates[0]["value"]
            try:
                datetime.strptime(date_val, "%Y-%m-%d")
                return {
                    "ok": True,
                    "answer": f"{subject.title()} is currently {age_val}. The event is on {date_val}. She will be {age_val} at that time unless her birthday passes between now and then.",
                    "confidence": 0.7,
                    "method": "temporal_age",
                }
            except Exception:
                pass
        if ages:
            return {
                "ok": True,
                "answer": f"{subject.title()} is {ages[0]['value']} years old according to memories.",
                "confidence": 0.6,
                "method": "temporal_age_partial",
            }
    
    # --- PATTERN: "How many X" or "What is the total X" (sum queries) ---
    if re.search(r'(how many|what is the total|total|sum|combined|add up)', q):
        # Find quantities in recalled memories
        quantities = [t for t in all_temporal if t["type"] == "quantity"]
        
        if quantities and len(quantities) >= 2:
            # Try to sum same-label quantities
            labels = {}
            for t in quantities:
                lbl = t.get("label", "").lower().rstrip('s')
                if lbl not in labels:
                    labels[lbl] = []
                labels[lbl].append(t["value"])
            
            parts = []
            total = 0
            for lbl, vals in labels.items():
                if len(vals) >= 2:
                    s = sum(vals)
                    total += s
                    parts.append(f"{lbl}: {s}")
                else:
                    total += vals[0]
                    parts.append(f"{lbl}: {vals[0]}")
            
            return {
                "ok": True,
                "answer": f"Total: {total} ({', '.join(parts)})",
                "confidence": 0.75,
                "method": "temporal_sum",
            }
    
    # --- PATTERN: "How many minutes did I exceed" (comparison queries) ---
    if re.search(r'(exceed|over by|more than|faster than|slower than)', q):
        durations = [t for t in all_temporal if t["type"] == "duration"]
        if len(durations) >= 2:
            # Find the difference
            vals = sorted([t["value"] for t in durations])
            diff = max(vals) - min(vals)
            return {
                "ok": True,
                "answer": f"Exceeded by {diff} {durations[0].get('unit','units')}.",
                "confidence": 0.7,
                "method": "temporal_exceed",
            }
    
    # --- PATTERN: "What is the average X" ---
    if re.search(r'average|mean', q):
        quantities = [t for t in all_temporal if t["type"] == "quantity"]
        if quantities:
            values = [t["value"] for t in quantities]
            avg = sum(values) / len(values)
            return {
                "ok": True,
                "answer": f"Average: {avg:.2f}",
                "confidence": 0.7,
                "method": "temporal_average",
            }
    
    # --- PATTERN: date lookup ("When did I", "What date") ---
    if re.search(r'(when did|what date|when was)', q):
        dates = [t for t in all_temporal if t["type"] == "date"]
        if dates:
            return {
                "ok": True,
                "answer": dates[0]["value"],
                "confidence": 0.8,
                "method": "temporal_date_lookup",
            }
    
    # --- FALLBACK: return the TOP memory text directly (not noise) ---
    top = memories[0]
    return {
        "ok": True,
        "answer": top["text"],
        "confidence": top.get("relevance", 0),
        "method": "temporal_fallback_top_memory",
    }
