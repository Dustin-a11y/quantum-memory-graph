"""The PCE optimiser must actually run, and the pipeline must say what ran.

`_build_pce_qaoa_circuit` referenced a bare `K` that was not a parameter, local
or global, so every `optimize_subgraph_pce` call raised NameError. The pipeline
caught it and fell back to greedy while still reporting `method="qaoa_subgraph"`,
which hid the failure completely. See issue #27.

These tests pin the two halves: the circuit builds, and the response exposes the
selection path that really executed.
"""
import numpy as np
import pytest

from quantum_memory_graph.pce_optimizer import optimize_subgraph_pce


def _instance(m=20, seed=0):
    rng = np.random.default_rng(seed)
    rel = rng.random(m)
    adj = (rng.random((m, m)) > 0.7).astype(float)
    return rel, np.maximum(adj, adj.T)


def test_pce_optimiser_does_not_raise():
    rel, adj = _instance()
    result = optimize_subgraph_pce(relevance_scores=rel, adjacency=adj, K=5)
    assert isinstance(result, dict)
    assert "selection" in result
    assert "score" in result


def test_pce_optimiser_returns_a_k_sized_selection():
    rel, adj = _instance(m=24, seed=3)
    result = optimize_subgraph_pce(relevance_scores=rel, adjacency=adj, K=4)
    assert len(result["selection"]) == 4
    assert all(0 <= i < 24 for i in result["selection"])
    assert len(set(result["selection"])) == 4


@pytest.mark.parametrize("K", [1, 3, 5])
def test_pce_optimiser_honours_the_budget_for_any_k(K):
    rel, adj = _instance(m=18, seed=K)
    result = optimize_subgraph_pce(relevance_scores=rel, adjacency=adj, K=K)
    assert len(result["selection"]) == K


def test_pipeline_reports_the_selection_path_that_ran(monkeypatch):
    """A greedy fallback must be visible as greedy, not as qaoa_subgraph."""
    from quantum_memory_graph import pipeline as P
    from quantum_memory_graph.graph import MemoryGraph

    g = MemoryGraph()
    for i in range(6):
        g.add_memory(f"memory number {i} about scheduling and rotations",
                     source=f"src{i}")

    real = P.recall("what about the rotation schedule", K=3, graph=g, method="qaoa")
    assert "selection_method" in real
    assert real["selection_method"] in {"qaoa", "pce_qaoa", "greedy_fallback",
                                        "greedy_safety_net", "trivial", "empty"}

    def boom(*a, **kw):
        raise NameError("name 'K' is not defined")

    monkeypatch.setattr(P, "_std_optimize_subgraph", boom)
    monkeypatch.setattr("quantum_memory_graph.pce_optimizer.optimize_subgraph_pce", boom)
    fell_back = P.recall("what about the rotation schedule", K=3, graph=g, method="qaoa")
    assert fell_back["selection_method"] == "greedy_safety_net"
    assert fell_back["qaoa_vs_greedy_pct"] is None
