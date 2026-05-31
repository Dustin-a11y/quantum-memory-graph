#!/usr/bin/env python3
"""
Synergy-aware reranker — word-overlap synergy + diversity selection.

Uses token-level overlap analysis to select chunks that are
complementary to each other, not just individually relevant.

DK 🦍
"""
import math
import numpy as np
from collections import defaultdict

STOP_WORDS = frozenset({
    "the","is","a","an","and","or","but","in","on","at",
    "to","for","of","with","by","from","was","were","are",
    "be","been","being","have","has","had","do","does","did",
    "will","would","could","should","may","might","can","this",
    "that","these","those","it","its","not","no","he","she",
    "his","her","my","me","i","you","we","us","they","them",
    "what","who","how","when","where","which",
})


def _tokenize(text):
    words = set()
    for w in text.lower().split():
        w = "".join(c for c in w if c.isalnum())
        if len(w) > 2 and w not in STOP_WORDS:
            words.add(w)
    return words


def _synergy_matrix(texts, query):
    """Pairwise synergy between chunks given a query."""
    n = len(texts)
    qt = _tokenize(query)
    mts = [_tokenize(t) for t in texts]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            qi = qt & mts[i]
            qj = qt & mts[j]
            combined = qi | qj
            complementary = (len(combined) - max(len(qi), len(qj))) / len(qt) if qt else 0.0
            mi, mj = mts[i], mts[j]
            u = mi | mj
            jaccard = len(mi & mj) / len(u) if u else 0.0
            relatedness = math.exp(-((jaccard - 0.2) ** 2) / 0.05)
            shared = (mi & mj) - qt
            bridge = min(len(shared) / 5.0, 0.3)
            synergy = max(0.0, complementary * 0.5 + relatedness * 0.3 + bridge * 0.2)
            mat[i][j] = mat[j][i] = synergy
    return mat


def _diversity_matrix(texts):
    """1 - Jaccard overlap between chunk token sets."""
    n = len(texts)
    mts = [_tokenize(t) for t in texts]
    mat = np.ones((n, n))
    np.fill_diagonal(mat, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            u = mts[i] | mts[j]
            overlap = len(mts[i] & mts[j]) / len(u) if u else 0.0
            mat[i][j] = mat[j][i] = 1.0 - overlap
    return mat


def select(cosine_scores, chunk_texts, query, K=5):
    """
    Select K chunks using synergy-aware greedy selection.

    Args:
        cosine_scores: 1D array of cosine scores
        chunk_texts: list of chunk text strings
        query: query text
        K: number of chunks to select

    Returns:
        List of selected chunk indices in selection order
    """
    n = len(cosine_scores)
    if n <= K:
        return list(range(n))

    synergy = _synergy_matrix(chunk_texts, query)
    diversity = _diversity_matrix(chunk_texts)

    selected = []
    remaining = set(range(n))
    first = int(np.argmax(cosine_scores))
    selected.append(first)
    remaining.remove(first)

    for _ in range(K - 1):
        best_idx, best_score = -1, -np.inf
        for i in remaining:
            if selected:
                avg_syn = float(np.mean([synergy[i][j] for j in selected]))
                avg_div = float(np.mean([diversity[i][j] for j in selected]))
            else:
                avg_syn = avg_div = 0.0
            score = 0.4 * cosine_scores[i] + 0.3 * avg_syn + 0.2 * avg_div + 0.1
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def rerank(cosine_scores, chunk_texts, chunk_session_ids, query, K=5):
    """
    Full synergy rerank: select chunks, rank sessions by contribution.

    Args:
        cosine_scores: per-chunk cosine scores
        chunk_texts: per-chunk text
        chunk_session_ids: per-chunk session ID
        query: query text
        K: number of chunks to select

    Returns:
        List of session IDs ranked by synergy contribution
    """
    selected = select(cosine_scores, chunk_texts, query, K)
    counts = defaultdict(int)
    for idx in selected:
        counts[chunk_session_ids[idx]] += 1
    return sorted(counts.keys(), key=lambda s: -counts[s])
