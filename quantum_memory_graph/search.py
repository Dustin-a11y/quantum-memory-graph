"""Hybrid text + embedding search for QMG."""

import numpy as np
from typing import List, Dict
from collections import defaultdict

# Simple BM25-inspired token scoring (no external deps needed)
from datetime import datetime, timezone


def tokenize(text: str) -> List[str]:
    """Extract meaningful tokens from text."""
    words = text.lower().split()
    tokens = []
    for w in words:
        w = ''.join(c for c in w if c.isalnum())
        if len(w) > 2:
            tokens.append(w)
    return tokens

def bm25_score(query: str, documents: List[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """Simple BM25-like scoring. Returns scores array."""
    doc_tokens = [tokenize(d) for d in documents]
    query_tokens = tokenize(query)
    
    if not query_tokens:
        return np.zeros(len(documents))
    
    # IDF
    N = len(documents)
    df = defaultdict(int)
    for dt in doc_tokens:
        for token in set(dt):
            df[token] += 1
    
    avgdl = np.mean([len(dt) for dt in doc_tokens]) if doc_tokens else 1
    
    scores = np.zeros(N)
    for token in query_tokens:
        if df[token] == 0:
            continue
        idf = np.log((N - df[token] + 0.5) / (df[token] + 0.5) + 1.0)
        for i, dt in enumerate(doc_tokens):
            tf = dt.count(token)
            if tf == 0:
                continue
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(dt) / avgdl))
            scores[i] += score
    
    # Normalize
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores

def _parse_ts(value):
    """Parse an ISO timestamp into a naive UTC datetime. Bad input raises ValueError."""
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = value
    else:
        try:
            ts = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("unparseable timestamp: %r" % (value,)) from exc
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def hybrid_search(query: str, graph, k: int = 5, text_weight: float = 0.3, embed_weight: float = 0.7,
                  time_range: Dict = None, recency_weight: float = 0.0) -> List[Dict]:
    """Hybrid search: BM25 text scoring + embedding similarity.

    Three optional time behaviours, all default-off so every existing caller is byte-identical:

    * ``time_range`` — hard filter, ``{"from": iso, "to": iso}``, both bounds optional and
      inclusive. A memory with no timestamp cannot be shown to fall inside a window, so it is
      excluded rather than guessed at.
    * ``recency_weight`` — soft bonus ``w * exp(-age_days / 30)`` added to the score. Weight is
      the ceiling of the bonus, so a caller can bound the influence.
    * every result carries ``timestamp`` and ``age_days`` so the reader can reason about time.

    Chronology is a retrieval input here, never a re-ordering of the injected block: sorting
    retrieved memories by date evicts the most relevant one from the window (measured -3.76pp).
    """
    lo = _parse_ts((time_range or {}).get("from"))
    hi = _parse_ts((time_range or {}).get("to"))
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("time_range: from is after to")
    if not graph.memories:
        return []
    
    mem_ids = list(graph.memories.keys())
    if time_range:
        kept = []
        for mid in mem_ids:
            ts = getattr(graph.memories[mid], "timestamp", None)
            if ts is None:
                continue
            if ts.tzinfo is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            if lo is not None and ts < lo:
                continue
            if hi is not None and ts > hi:
                continue
            kept.append(mid)
        mem_ids = kept
        if not mem_ids:
            return []
    texts = [graph.memories[mid].text for mid in mem_ids]
    
    # Text scores
    text_scores = bm25_score(query, texts)
    
    # Embedding scores
    query_emb = graph.embedder.encode([query], normalize_embeddings=True)[0]
    embed_scores = np.zeros(len(mem_ids))
    for i, mid in enumerate(mem_ids):
        mem = graph.memories[mid]
        if mem.embedding is not None:
            embed_scores[i] = float(np.dot(query_emb, mem.embedding))
    
    # Normalize embedding scores
    if embed_scores.max() > 0:
        embed_scores = embed_scores / embed_scores.max()
    
    # Combine
    combined = text_weight * text_scores + embed_weight * embed_scores

    # Optional recency bonus: bounded by recency_weight, decaying with a 30 day half-life-ish scale.
    now = datetime.utcnow()
    ages = []
    for mid in mem_ids:
        ts = getattr(graph.memories[mid], "timestamp", None)
        if ts is None:
            ages.append(None)
            continue
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        ages.append(max(0.0, (now - ts).total_seconds() / 86400.0))
    if recency_weight:
        # Multiplicative, because the combined score is not on a fixed scale: bm25_score is an
        # unbounded IDF sum while the embedding term is normalised to <= 1, so an additive bonus of
        # 0.1-0.2 is invisible whenever text dominates and only breaks exact ties (measured: 0
        # genuine overtakes below w=1.0). A relative bonus of w * decay is meaningful on any scale
        # and still leaves recency_weight=0 byte-identical.
        decay = np.array([
            0.0 if age is None else float(np.exp(-age / 30.0))
            for age in ages
        ])
        combined = combined * (1.0 + recency_weight * decay)
    
    # Top K
    top_indices = np.argsort(combined)[::-1][:k]
    
    results = []
    for idx in top_indices:
        mid = mem_ids[idx]
        mem = graph.memories[mid]
        _ts = getattr(mem, "timestamp", None)
        if _ts is not None and _ts.tzinfo is not None:
            _ts = _ts.astimezone(timezone.utc).replace(tzinfo=None)
        results.append({
            "id": mid,
            "text": mem.text,
            "entities": mem.entities,
            "source": mem.source,
            "timestamp": _ts.isoformat() if _ts is not None else None,
            "age_days": (
                None if _ts is None
                else round(max(0.0, (now - _ts).total_seconds() / 86400.0), 3)
            ),
            "text_score": round(float(text_scores[idx]), 4),
            "embed_score": round(float(embed_scores[idx]), 4),
            "combined_score": round(float(combined[idx]), 4),
        })
    
    return results
