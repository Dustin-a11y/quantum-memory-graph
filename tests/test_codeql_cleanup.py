import pickle

import networkx as nx
import numpy as np

from quantum_memory_graph.graph import Memory, MemoryGraph


def _legacy_graph_path(tmp_path):
    graph = nx.Graph()
    graph.add_node("m1")
    memories = {
        "m1": Memory(id="m1", text="hello world", entities=["world"])
    }
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as handle:
        pickle.dump({"G": graph, "memories": memories}, handle)
    return path


def test_load_legacy_graph_rebuilds_indices(tmp_path):
    loaded = MemoryGraph.load(str(_legacy_graph_path(tmp_path)))

    assert isinstance(loaded, MemoryGraph)
    assert "m1" in loaded.memories
    assert loaded._entity_index["world"] == {"m1"}


def test_load_legacy_graph_honors_model_override(tmp_path):
    loaded = MemoryGraph.load(
        str(_legacy_graph_path(tmp_path)), model="thenlper/gte-large"
    )

    assert loaded._model_name == "thenlper/gte-large"


def test_subgraph_optimizer_qaoa_path_remains_available():
    from quantum_memory_graph.subgraph_optimizer import optimize_subgraph

    relevance = np.array([0.9, 0.8, 0.7])
    adjacency = np.zeros((3, 3))
    result = optimize_subgraph(relevance, adjacency, K=2, grid_size=1, shots=32)

    assert result["method"] in {"qaoa", "greedy_fallback"}
    assert len(result["selection"]) == 2
