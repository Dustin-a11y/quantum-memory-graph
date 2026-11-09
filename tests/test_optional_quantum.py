"""The light install must be importable, and QAOA must degrade to greedy when qiskit is absent.

`qiskit`/`qiskit-aer` are the optional `quantum` extra, but `subgraph_optimizer` imported them at
module scope and `__init__` imports that module — so `pip install quantum-memory-graph` (core only)
produced a package that raised ModuleNotFoundError on import. These tests pin the guard so the extra
can never silently become mandatory again.
"""

import pytest

from quantum_memory_graph import subgraph_optimizer as so


def test_qiskit_availability_flag_exists():
    assert isinstance(so.QISKIT_AVAILABLE, bool)


def test_optimizer_works_without_qiskit(monkeypatch):
    """Force the light-install state and confirm the classical fallback still answers."""
    monkeypatch.setattr(so, "QISKIT_AVAILABLE", False)
    monkeypatch.setattr(so, "QuantumCircuit", None)
    monkeypatch.setattr(so, "AerSimulator", None)

    import numpy as np

    rel = np.array([0.9, 0.7, 0.4, 0.2, 0.1])
    adj = np.ones((5, 5)) - np.eye(5)

    result = so.optimize_subgraph(rel, adj, K=3)

    assert len(result["selection"]) == 3
    # without qiskit the QAOA branch must not be the method that produced the selection
    assert result["method"] in ("greedy_fallback", "trivial")
    assert "greedy" in result and len(result["greedy"]["selection"]) == 3


def test_trivial_case_never_needs_qiskit(monkeypatch):
    monkeypatch.setattr(so, "QISKIT_AVAILABLE", False)
    import numpy as np

    result = so.optimize_subgraph(np.array([0.5, 0.4]), np.ones((2, 2)), K=5)
    assert result["method"] == "trivial"
