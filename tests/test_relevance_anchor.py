"""The returned memories must contain the most relevant candidate.

Measured on LongMemEval (issue #27): plain embedding search ranked the gold
session first, and the unanchored subgraph optimiser then dropped it in 55% of
questions. Anchoring is the smallest change that makes selection non-destructive.
"""
import numpy as np
import pytest

from quantum_memory_graph.pipeline import _anchor_top_relevance


def test_anchor_keeps_the_top_candidate_when_optimiser_dropped_it():
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.4])
    # optimiser chose nothing relevant — the pattern that lost 55% of retrievals
    assert _anchor_top_relevance([4, 2, 1], scores, K=3, anchor=1)[0] == 0


def test_anchor_leaves_a_correct_selection_alone():
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert _anchor_top_relevance([0, 1, 2], scores, K=3, anchor=1) == [0, 1, 2]


def test_result_is_always_k_long_and_unique():
    scores = np.array([0.5, 0.9, 0.1, 0.7, 0.3, 0.2])
    sel = _anchor_top_relevance([5, 5, 2], scores, K=4, anchor=1)
    assert len(sel) == 4
    assert len(set(sel)) == 4


def test_optimiser_picks_fill_the_remaining_slots_after_the_anchor():
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.4])
    sel = _anchor_top_relevance([3, 4], scores, K=3, anchor=1)
    assert sel[0] == 0            # anchor
    assert sel[1:] == [3, 4]      # optimiser's own choices preserved


def test_shortfall_is_filled_in_relevance_order():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    sel = _anchor_top_relevance([3], scores, K=3, anchor=1)
    assert sel == [0, 3, 1]


@pytest.mark.parametrize("anchor", [2, 3])
def test_larger_anchors_keep_more_of_the_top(anchor):
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.7])
    sel = _anchor_top_relevance([0], scores, K=3, anchor=anchor)
    assert set([2, 3, 4][:anchor]).issubset(set(sel))


def test_anchor_zero_preserves_historic_behaviour():
    scores = np.array([0.9, 0.1, 0.2])
    assert _anchor_top_relevance([2, 1], scores, K=2, anchor=0) == [2, 1]


def test_anchor_k_is_pure_relevance_order():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert _anchor_top_relevance([0, 2], scores, K=3, anchor=3) == [1, 3, 2]


def test_degenerate_inputs_are_safe():
    assert _anchor_top_relevance([], np.array([]), K=5) == []
    assert _anchor_top_relevance([0], np.array([0.4]), K=0) == []
    # K larger than the pool
    assert _anchor_top_relevance([1], np.array([0.2, 0.9]), K=10) == [1, 0]


def test_out_of_range_optimiser_indices_are_ignored():
    scores = np.array([0.5, 0.9, 0.3])
    sel = _anchor_top_relevance([99, -1, 2], scores, K=2, anchor=1)
    assert sel == [1, 2]


def test_recall_exposes_the_selection_path_and_anchor():
    from quantum_memory_graph.graph import MemoryGraph
    from quantum_memory_graph import pipeline as P

    g = MemoryGraph()
    for i in range(6):
        g.add_memory(f"note {i} about the rotation schedule and staffing",
                     source=f"s{i}")
    out = P.recall("rotation schedule staffing", K=3, graph=g, method="qaoa")
    assert out["selection_method"] in {"qaoa", "pce_qaoa", "greedy_fallback",
                                       "greedy_safety_net", "trivial", "empty"}
    assert out["relevance_anchor"] == 1
    assert len(out["memories"]) <= 3


def test_recall_returns_the_top_relevance_memory_even_if_the_optimiser_fails(monkeypatch):
    """A broken optimiser must not cost us the best candidate."""
    from quantum_memory_graph.graph import MemoryGraph
    from quantum_memory_graph import pipeline as P

    g = MemoryGraph()
    for i in range(6):
        g.add_memory(f"note {i} about the rotation schedule and staffing",
                     source=f"s{i}")

    def worst_possible(relevance_scores, adjacency, K, alpha=0.4, beta_conn=0.35,
                       gamma_cov=0.25, **kw):
        # deliberately returns the worst candidates
        order = list(np.argsort(np.asarray(relevance_scores)))[:K]
        return {"selection": [int(i) for i in order], "score": 0.0,
                "greedy": {"selection": [int(i) for i in order], "score": 0.0},
                "method": "adversarial"}

    monkeypatch.setattr(P, "_std_optimize_subgraph", worst_possible)
    monkeypatch.setattr("quantum_memory_graph.pce_optimizer.optimize_subgraph_pce",
                        worst_possible)

    out = P.recall("rotation schedule staffing", K=3, graph=g, method="qaoa")
    assert out["selection_method"] == "adversarial"
    ids = [m["id"] for m in out["memories"]]
    best = max(g.memories.values(),
               key=lambda m: float(np.dot(g._embedder.encode(["rotation schedule staffing"])[0],
                                          m.embedding)))
    assert best.id in ids, "anchoring must keep the top-relevance memory"


def test_anchor_zero_lets_the_adversarial_optimiser_through(monkeypatch):
    """Control: with anchoring off, the same optimiser does lose the best node."""
    from quantum_memory_graph.graph import MemoryGraph
    from quantum_memory_graph import pipeline as P

    g = MemoryGraph()
    for i in range(6):
        g.add_memory(f"note {i} about the rotation schedule and staffing",
                     source=f"s{i}")

    def worst_possible(relevance_scores, adjacency, K, alpha=0.4, beta_conn=0.35,
                       gamma_cov=0.25, **kw):
        order = list(np.argsort(np.asarray(relevance_scores)))[:K]
        return {"selection": [int(i) for i in order], "score": 0.0,
                "method": "adversarial"}

    monkeypatch.setattr(P, "_std_optimize_subgraph", worst_possible)
    monkeypatch.setattr("quantum_memory_graph.pce_optimizer.optimize_subgraph_pce",
                        worst_possible)

    out = P.recall("rotation schedule staffing", K=3, graph=g, method="qaoa",
                   relevance_anchor=0)
    best = max(g.memories.values(),
               key=lambda m: float(np.dot(g._embedder.encode(["rotation schedule staffing"])[0],
                                          m.embedding)))
    assert best.id not in [m["id"] for m in out["memories"]]
