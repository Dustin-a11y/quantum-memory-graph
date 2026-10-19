"""Hybrid text + embedding search for QMG."""

import numpy as np
from typing import List, Dict
from collections import defaultdict

# Simple BM25-inspired token scoring (no external deps needed)
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

def hybrid_search(query: str, graph, k: int = 5, text_weight: float = 0.3, embed_weight: float = 0.7) -> List[Dict]:
    """Hybrid search: BM25 text scoring + embedding similarity."""
    if not graph.memories:
        return []
    
    mem_ids = list(graph.memories.keys())
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
    
    # Top K
    top_indices = np.argsort(combined)[::-1][:k]
    
    results = []
    for idx in top_indices:
        mid = mem_ids[idx]
        mem = graph.memories[mid]
        results.append({
            "id": mid,
            "text": mem.text,
            "entities": mem.entities,
            "source": mem.source,
            "text_score": round(float(text_scores[idx]), 4),
            "embed_score": round(float(embed_scores[idx]), 4),
            "combined_score": round(float(combined[idx]), 4),
        })
    
    return results
