"""kong_graph must be a transparent alias, never a fork.

Every persisted graph pair pickles class references from
``quantum_memory_graph.graph``, ``.semantic`` and ``.learning``, so the old
module name has to stay importable forever. These tests hold the alias honest:
same objects, working submodule imports, and old-name pickles still loading.
"""
import pickle

import pytest


def test_package_aliases_the_same_module():
    import kong_graph
    import quantum_memory_graph

    assert kong_graph.__version__ == quantum_memory_graph.__version__
    assert kong_graph.__all__ == list(quantum_memory_graph.__all__)
    assert kong_graph.MemoryGraph is quantum_memory_graph.MemoryGraph


def test_submodule_import_resolves_to_the_same_object():
    from kong_graph.graph import MemoryGraph as alias_cls
    from quantum_memory_graph.graph import MemoryGraph as real_cls

    assert alias_cls is real_cls


def test_the_three_pickled_classes_resolve_through_the_alias():
    """The exact three modules every pair references must alias cleanly."""
    from kong_graph.graph import MemoryGraph
    from kong_graph.learning import RecallTracker
    from kong_graph.semantic import SemanticStore

    from quantum_memory_graph import graph as real_graph
    from quantum_memory_graph import learning as real_learning
    from quantum_memory_graph import semantic as real_semantic

    assert MemoryGraph is real_graph.MemoryGraph
    assert RecallTracker is real_learning.RecallTracker
    assert SemanticStore is real_semantic.SemanticStore


def test_importing_the_alias_does_not_eagerly_import_optional_extras():
    """kong_graph.api pulls fastapi; importing kong_graph must not force it."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        import kong_graph
        assert "quantum_memory_graph.api" not in sys.modules, "api imported eagerly"
        print("ok")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-1500:]
    assert "ok" in out.stdout


def test_object_created_through_the_alias_is_the_real_class():
    import kong_graph
    import quantum_memory_graph

    g = kong_graph.MemoryGraph()
    assert isinstance(g, quantum_memory_graph.MemoryGraph)

    g.add_memory("The Kong Graph alias must not fork the class hierarchy.",
                 source="alias-test")
    assert len(g.memories) == 1


def test_a_pickle_naming_the_old_module_still_loads(tmp_path):
    """A pair written by the old name must keep loading after the rebrand."""
    from quantum_memory_graph.graph import MemoryGraph

    g = MemoryGraph()
    g.add_memory("Pickles reference quantum_memory_graph, not kong_graph.",
                 source="alias-pickle-test")
    path = tmp_path / "graph.pkl"
    with open(path, "wb") as f:
        pickle.dump(g, f)

    raw = path.read_bytes()
    assert b"quantum_memory_graph.graph" in raw, "fixture does not exercise the old name"

    with open(path, "rb") as f:
        reloaded = pickle.load(f)
    assert isinstance(reloaded, MemoryGraph)
    assert len(reloaded.memories) == len(g.memories)


@pytest.mark.parametrize("name", ["graph", "semantic", "learning", "recency", "pipeline"])
def test_every_core_submodule_is_reachable_through_the_alias(name):
    import importlib

    alias = importlib.import_module(f"kong_graph.{name}")
    real = importlib.import_module(f"quantum_memory_graph.{name}")
    assert alias is real
