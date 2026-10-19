"""Old-generation pickles must converge on the strict pair validator.

Third-generation packages wrote ``sources``, ``_source_index`` and
``max_connections`` onto ``MemoryGraph``. No current code reads them, but
``rollback._load_pair_checked`` rejects any unknown attribute, so a graph that
``load()`` did not strip was re-saved in a state its own validator refused.
These tests pin the guarded strip: exactly those three names, only when they
carry no data, and nothing else.
"""

import pickle
from datetime import datetime

import pytest

from quantum_memory_graph import rollback
from quantum_memory_graph.graph import Memory, MemoryGraph

_TS = datetime(2026, 1, 2, 3, 4, 5)
_LEGACY = ("sources", "_source_index", "max_connections")


def _build_graph(pairs=(("DK keeps the pair validator strict", "unit-test"),
                        ("Vestigial attributes carry no data", ""))):
    """Small synthetic graph: no embedding model is ever constructed."""
    graph = MemoryGraph(similarity_threshold=0.3)
    for text, source in pairs:
        memory = Memory(
            id=Memory.make_id(text), text=text, embedding=None, entities=[],
            timestamp=_TS, source=source, metadata={}, temporal_entities=[],
        )
        graph.memories[memory.id] = memory
        graph.graph.add_node(memory.id, memory=memory)
    graph._rebuild_indices()
    return graph


def _add_legacy_attrs(graph, sources=None, source_index=None,
                      max_connections=50):
    graph.sources = {} if sources is None else sources
    graph._source_index = {} if source_index is None else source_index
    graph.max_connections = max_connections
    return graph


def _write_pair(directory, graph):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "graph.pkl", "wb") as fh:
        pickle.dump(graph, fh)
    graph.save_json(str(directory / "graph.json"))
    return directory


def test_load_strips_legacy_attributes(tmp_path):
    legacy = _add_legacy_attrs(_build_graph())
    pair = _write_pair(tmp_path / "legacy", legacy)
    assert set(vars(legacy)) - set(rollback._GRAPH_ATTRS) == set(_LEGACY)

    loaded = MemoryGraph.load(str(pair / "graph.pkl"))

    for attr in _LEGACY:
        assert not hasattr(loaded, attr), attr
    assert set(vars(loaded)) == set(rollback._GRAPH_ATTRS)


def test_round_trip_through_save_is_validator_clean(tmp_path):
    """Regression: the pre-fix pair is rejected; load+save must clear it."""
    legacy = _add_legacy_attrs(_build_graph())
    old_pair = _write_pair(tmp_path / "legacy", legacy)
    with pytest.raises(ValueError, match="unknown graph attributes"):
        rollback._load_pair_checked(str(old_pair), "legacy pair")

    loaded = MemoryGraph.load(str(old_pair / "graph.pkl"))
    migrated = tmp_path / "migrated"
    migrated.mkdir()
    loaded.save(str(migrated / "graph.pkl"))
    loaded.save_json(str(migrated / "graph.json"))

    graph, doc, _pkl, _json, _digests = rollback._load_pair_checked(
        str(migrated), "migrated pair")
    assert set(vars(graph)) == set(rollback._GRAPH_ATTRS)
    assert set(graph.memories) == set(legacy.memories)
    assert doc["node_count"] == len(graph.memories)
    assert graph.memories == legacy.memories
    assert set(graph.graph.edges) == set(legacy.graph.edges)


def test_non_empty_sources_refuses_to_drop(tmp_path):
    legacy = _add_legacy_attrs(_build_graph(), sources={"doc": ["m1"]})
    pair = _write_pair(tmp_path / "legacy", legacy)

    with pytest.raises(ValueError, match="non-empty"):
        MemoryGraph.load(str(pair / "graph.pkl"))


def test_non_empty_source_index_refuses_to_drop(tmp_path):
    legacy = _add_legacy_attrs(_build_graph(), source_index={"m1": "doc"})
    pair = _write_pair(tmp_path / "legacy", legacy)

    with pytest.raises(ValueError, match="non-empty"):
        MemoryGraph.load(str(pair / "graph.pkl"))


def test_non_int_max_connections_refuses_to_drop(tmp_path):
    legacy = _add_legacy_attrs(_build_graph(), max_connections="50")
    pair = _write_pair(tmp_path / "legacy", legacy)

    with pytest.raises(ValueError, match="max_connections"):
        MemoryGraph.load(str(pair / "graph.pkl"))


def test_modern_pair_loads_unchanged(tmp_path):
    modern = _build_graph()
    pair = _write_pair(tmp_path / "modern", modern)
    before = set(vars(modern))
    assert before == set(rollback._GRAPH_ATTRS)
    rollback._load_pair_checked(str(pair), "modern pair")

    loaded = MemoryGraph.load(str(pair / "graph.pkl"))

    assert set(vars(loaded)) == before
    assert loaded.memories == modern.memories
    rollback._load_pair_checked(str(pair), "modern pair")


def test_unknown_attributes_are_not_stripped(tmp_path):
    """Only the three legacy names are removed; future fields still fail."""
    future = _build_graph()
    future.future_field = {"typed": True}
    pair = _write_pair(tmp_path / "future", future)

    loaded = MemoryGraph.load(str(pair / "graph.pkl"))

    assert loaded.future_field == {"typed": True}
    with pytest.raises(ValueError, match="unknown graph attributes"):
        rollback._load_pair_checked(str(pair), "future pair")
