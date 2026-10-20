"""Kong Graph — the public name for this package.

``quantum_memory_graph`` remains the importable module name and always will be:
every persisted graph pair pickles class references such as
``quantum_memory_graph.graph.MemoryGraph``,
``quantum_memory_graph.semantic.SemanticStore`` and
``quantum_memory_graph.learning.RecallTracker``. Renaming the package outright
would make every existing pair unloadable.

So this is a transparent alias, not a copy: ``kong_graph`` and
``quantum_memory_graph`` resolve to the *same* module objects, so
``kong_graph.graph.MemoryGraph is quantum_memory_graph.graph.MemoryGraph``.
Class identity, isinstance() checks and pickle round-trips keep working through
either name.

Usage::

    import kong_graph
    g = kong_graph.MemoryGraph()

    from kong_graph.graph import MemoryGraph       # same object
    from kong_graph.semantic import SemanticStore  # same object

Migrating a call site to the new name is safe at any time and does not require
the old name to be retired first.
"""
from __future__ import annotations

import importlib
import sys

import quantum_memory_graph as _impl

__all__ = list(_impl.__all__)
__version__ = _impl.__version__

_OLD = _impl.__name__

# Registered eagerly so `from kong_graph.X import Y` resolves to the identical
# module object. `api` is excluded on purpose: it imports fastapi, which is an
# optional extra, so importing the alias must not force it.
_EAGER = (
    "graph",
    "learning",
    "semantic",
    "search",
    "recency",
    "pipeline",
    "subgraph_optimizer",
    "pce_optimizer",
    "synergy_reranker",
    "admission",
    "cache_attestation",
    "inventory_v2",
    "rollback",
)

for _name in _EAGER:
    try:
        sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"{_OLD}.{_name}")
    except ImportError:  # pragma: no cover - a submodule missing in an old release
        continue
del _name


def __getattr__(name):
    """Forward attributes, including lazily imported submodules (e.g. ``api``)."""
    if name.startswith("__"):
        raise AttributeError(name)
    try:
        return getattr(_impl, name)
    except AttributeError:
        pass
    module = importlib.import_module(f"{_OLD}.{name}")
    sys.modules[f"{__name__}.{name}"] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)) | set(_EAGER))
