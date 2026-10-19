"""Strict, versioned QMG state inventory for DK maintenance receipts.

Only digests and counts leave this module. Unsupported or unaccounted persisted
state is a refusal, never silently omitted. Legacy v1 is retained separately in
admission.py and MUST NOT be used as a full-state proof.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from datetime import datetime, timedelta
from typing import Any

import networkx as nx
import numpy as np

SCHEMA = "qmg-graph-inventory-v2"
_FROZEN_LEGACY_GRAPH_SHA256 = "d161b1b0fdea69302ca4b49548b7541fd2a043885a3bb85e94016785daa2757c"
_GRAPH_FIELDS = frozenset({
    "graph", "memories", "similarity_threshold", "_model_name", "_embedder",
    "_is_bge", "_bm25", "_bm25_corpus", "_entity_index", "semantic_store",
    "recall_tracker", "learned_weights",
})
_STORE_FIELDS = frozenset({"facts", "by_subject_relation", "labels_by_id", "label_index"})
_TRACKER_FIELDS = frozenset({"history", "graph_dir", "co_recall_count"})
_NX_FIELDS = frozenset({
    "graph", "_node", "_adj", "__networkx_cache__", "nodes", "edges",
    "degree", "adj",
})


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _string(value: Any, path: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{path}: expected string")
    value.encode("utf-8", "strict")
    return value


def _finite(value: Any, path: str) -> int | float:
    if type(value) not in (float, int) or type(value) is bool or not math.isfinite(value):
        raise ValueError(f"{path}: expected finite number")
    # Return the scalar unchanged: the canonical JSON encoder used by _digest
    # keeps int 1 ("1") and float 1.0 ("1.0") distinct, while float(value)
    # would collapse both into one canonical row and hide a persisted type
    # change from sha256.full. NaN/inf are rejected above, and _digest also
    # serializes with allow_nan=False.
    return value


def _time(value: Any, path: str, *, aware: bool = False) -> dict | None:
    """Tag a persisted datetime: canonical ISO, fold and tzinfo identity.

    ``isoformat()`` loses ``datetime.fold`` and the aware-UTC tzinfo class, so
    a fold-only flip or a ``timezone.utc`` -> ``ZoneInfo("UTC")`` swap is real
    persisted state with an identical ISO string. Both are encoded here; the
    naive branch keeps tzinfo absent and an aware non-UTC offset is refused.
    """
    if value is None:
        return None
    if type(value) is not datetime:
        raise ValueError(f"{path}: expected datetime")
    if aware:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError(f"{path}: expected aware UTC")
        tzinfo = value.tzinfo
        try:
            name = tzinfo.tzname(value)
        except Exception as exc:
            raise ValueError(f"{path}: tzinfo name is unavailable") from exc
        if name is not None and type(name) is not str:
            raise ValueError(f"{path}: malformed tzinfo name")
        return {"iso": value.isoformat(timespec="microseconds"),
                "fold": value.fold,
                "tzinfo": {"module": type(tzinfo).__module__,
                           "qualname": type(tzinfo).__qualname__,
                           "name": name}}
    if value.tzinfo is not None:
        # MemoryGraph.load normalizes aware legacy timestamps. Do not claim raw
        # aware-vs-naive byte identity from a loaded-state inventory.
        raise ValueError(f"{path}: unnormalized memory timestamp")
    return {"iso": value.isoformat(timespec="microseconds"),
            "fold": value.fold}


def _plain(value: Any, path: str, depth: int = 0) -> Any:
    """Tag container/scalar types, reject arbitrary objects and nonfinite data."""
    if depth > 40:
        raise ValueError(f"{path}: excessive nesting")
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str:
        return _string(value, path)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: nonfinite number")
        return value
    if isinstance(value, np.generic):
        scalar = value.item()
        if type(scalar) not in (bool, int, float):
            raise ValueError(f"{path}: unsupported numpy scalar")
        return {"numpy_dtype": value.dtype.str,
                "value": _plain(scalar, path, depth + 1)}
    if type(value) is datetime:
        return {"datetime": _time(value, path, aware=True)}
    if type(value) is list:
        return [_plain(v, f"{path}[{i}]", depth + 1) for i, v in enumerate(value)]
    if type(value) is tuple:
        return {"tuple": [_plain(v, f"{path}[{i}]", depth + 1)
                          for i, v in enumerate(value)]}
    if type(value) is dict:
        # Persisted insertion order is behavior-affecting state (metadata,
        # BM25 doc_freqs and node/edge/graph attributes); a plain dict
        # projection would drop it, so the mapping is tagged as an ordered
        # key/value row list. Keys are validated exactly as before.
        rows = []
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path}: mapping keys must be strings")
            rows.append([_string(key, path),
                         _plain(item, f"{path}.{key}", depth + 1)])
        return {"dict": rows}
    raise ValueError(f"{path}: unsupported persisted type {type(value).__name__}")


def _embedding(value: Any, path: str) -> Any:
    if value is None:
        return None
    if (type(value) is not np.ndarray or value.ndim != 1
            or value.dtype.kind != "f" or not value.flags.c_contiguous
            or not np.isfinite(value).all()):
        raise ValueError(f"{path}: unsupported or nonfinite embedding")
    return {"dtype": value.dtype.str, "shape": list(value.shape),
            "writeable": bool(value.flags.writeable),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def _pairs(mapping: Any, path: str) -> list:
    if type(mapping) is not dict:
        raise ValueError(f"{path}: expected dictionary")
    rows = []
    for pair, weight in mapping.items():
        if (type(pair) is not tuple or len(pair) != 2
                or any(type(item) is not str for item in pair)):
            raise ValueError(f"{path}: malformed tuple key")
        rows.append([_string(pair[0], path), _string(pair[1], path),
                     _finite(weight, path)])
    # Pair iteration order is persisted state: never column-sort it away. Two
    # mappings with identical items inserted in a different order are distinct
    # states (learning.py reads them back in exactly this order).
    return rows


def _set_index(mapping: Any, path: str) -> list:
    if type(mapping) is not dict:
        raise ValueError(f"{path}: expected dictionary")
    rows = []
    for key, values in mapping.items():
        if type(values) is not set:
            raise ValueError(f"{path}: expected exact set values")
        rows.append([_string(key, path), sorted(_string(v, path) for v in values)])
    rows.sort(key=lambda row: row[0])
    return rows


def _tuple_set_index(mapping: Any, path: str) -> list:
    if type(mapping) is not dict:
        raise ValueError(f"{path}: expected dictionary")
    rows = []
    for pair, values in mapping.items():
        if (type(pair) is not tuple or len(pair) != 2 or type(values) is not set):
            raise ValueError(f"{path}: malformed tuple/set index")
        rows.append([_string(pair[0], path), _string(pair[1], path),
                     sorted(_string(v, path) for v in values)])
    rows.sort(key=lambda row: row[:2])
    return rows


def _bm25(value: Any) -> Any:
    if value is None:
        return None
    if (type(value).__module__ != "rank_bm25"
            or type(value).__name__ != "BM25Okapi"):
        raise ValueError("_bm25: unsupported cache class")
    attrs = vars(value)
    expected = {"average_idf", "avgdl", "b", "corpus_size", "doc_freqs",
                "doc_len", "epsilon", "idf", "k1", "tokenizer"}
    if set(attrs) != expected or attrs["tokenizer"] is not None:
        raise ValueError("_bm25: unaccounted cache state")
    return {"class": "rank_bm25.BM25Okapi", "state": _plain(attrs, "_bm25")}


def _networkx_cache_check(g: nx.Graph) -> None:
    """Validate excluded NetworkX derived views, never accept altered views."""
    attrs = vars(g)
    if set(attrs) - _NX_FIELDS or not {"graph", "_node", "_adj"} <= set(attrs):
        raise ValueError("NetworkX graph has unknown/missing instance fields")
    if attrs.get("__networkx_cache__", {}) != {}:
        raise ValueError("NetworkX conversion cache is not empty")
    if "nodes" in attrs and getattr(attrs["nodes"], "_nodes", None) is not g._node:
        raise ValueError("NetworkX cached nodes view differs from raw nodes")
    if "edges" in attrs and getattr(attrs["edges"], "_adjdict", None) is not g._adj:
        # NetworkX versions use either _adjdict or _adj; do not trust a view
        # that does not expose the live graph's adjacency mapping.
        if getattr(attrs["edges"], "_adj", None) is not g._adj:
            raise ValueError("NetworkX cached edges view differs from adjacency")
    if "adj" in attrs and getattr(attrs["adj"], "_atlas", None) is not g._adj:
        raise ValueError("NetworkX cached adjacency view differs from raw adjacency")
    if "degree" in attrs and getattr(attrs["degree"], "_graph", None) is not g:
        raise ValueError("NetworkX cached degree view differs from graph")


def graph_inventory_v2(graph: Any, *, allow_legacy_untyped: bool = False) -> dict:
    """Canonical digest of the loaded, behavior-affecting DK graph state.

    Every supported Memory, Graph, semantic, index and learning field is
    represented. Runtime _embedder MUST be explicitly verified and detached
    before inventory: silently hashing only its name would miss changed model
    weights/tokenizer behavior. No raw text, tensor or query leaves the helper.
    """
    from .graph import Memory, MemoryGraph
    from .learning import RecallTracker
    from .semantic import SemanticFact, SemanticStore

    fields = set(vars(graph))
    old_untyped = (allow_legacy_untyped is True
                   and fields == _GRAPH_FIELDS - {"semantic_store"})
    if type(graph) is not MemoryGraph or (fields != _GRAPH_FIELDS and not old_untyped):
        raise ValueError("MemoryGraph has missing/unknown persisted fields")
    if old_untyped:
        from . import graph as graph_module
        path = os.path.realpath(graph_module.__file__)
        if path.endswith(".pyc") or not path.endswith(".py") or os.path.islink(graph_module.__file__):
            raise ValueError("frozen old graph import origin is not regular source")
        with open(path, "rb") as source:
            if hashlib.sha256(source.read()).hexdigest() != _FROZEN_LEGACY_GRAPH_SHA256:
                raise ValueError("legacy-untyped projection requires pinned frozen old graph.py")
    if graph._embedder is not None:
        raise ValueError("runtime embedder cache is not verified/detached")
    if type(graph.graph) is not nx.Graph or type(graph.memories) not in (dict,):
        raise ValueError("unsupported graph or memory mapping")
    nxg = graph.graph
    _networkx_cache_check(nxg)
    if set(nxg._node) != set(graph.memories) or set(nxg._adj) != set(nxg._node):
        raise ValueError("graph nodes disagree with memory inventory")
    memory_order = list(graph.memories)
    node_order = list(nxg._node)
    for mid in memory_order + node_order:
        _string(mid, "memory.id")
    memories = []
    derived_entities: dict[str, set[str]] = {}
    for mid in sorted(graph.memories):
        mem = graph.memories[mid]
        fields = set(vars(mem))
        current = set(Memory.__dataclass_fields__)
        # Three pre-temporal DK memories legitimately predate this field. Its
        # ABSENCE is explicit state, not silently converted to an empty list.
        if type(mem) is not Memory or fields not in (current, current - {"temporal_entities"}):
            raise ValueError("memory has missing/unknown persisted fields")
        if mem.id != mid or type(mem.entities) is not list:
            raise ValueError("malformed memory ID or entities")
        for entity in mem.entities:
            derived_entities.setdefault(_string(entity, "memory.entities"), set()).add(mid)
        row = {
            "id": _string(mem.id, "memory.id"),
            "text": _string(mem.text, "memory.text"),
            "source": _string(mem.source, "memory.source"),
            "embedding": _embedding(mem.embedding, "memory.embedding"),
            "entities": list(mem.entities),
            "timestamp": _time(mem.timestamp, "memory.timestamp"),
            "metadata": _plain(mem.metadata, "memory.metadata"),
            "temporal_entities": (_plain(vars(mem)["temporal_entities"], "memory.temporal_entities")
                                  if "temporal_entities" in fields else {"legacy_missing": True}),
        }
        if (type(mem.metadata) is not dict
                or ("temporal_entities" in fields
                    and type(vars(mem)["temporal_entities"]) is not list)):
            raise ValueError("memory metadata/temporal annotations are malformed")
        node = nxg._node[mid]
        if type(node) is not dict or node.get("memory") is not mem:
            raise ValueError("graph node does not alias its memory object")
        extras = {k: v for k, v in node.items() if k != "memory"}
        memories.append([mid, row, _plain(extras, "graph.node.attrs")])
    entities = _set_index(graph._entity_index, "_entity_index")
    if entities != _set_index(derived_entities, "derived_entities"):
        raise ValueError("entity index does not reconstruct from memories")
    if type(graph._bm25_corpus) is not list or graph._bm25_corpus != [
            mem.text for mem in graph.memories.values()]:
        raise ValueError("BM25 corpus does not reconstruct in memory insertion order")
    corpus = [_string(text, "_bm25_corpus") for text in graph._bm25_corpus]
    adjacency = []
    for mid in node_order:
        neighbors = list(nxg._adj[mid])
        for nid in neighbors:
            if (nid not in nxg._node or mid not in nxg._adj[nid]
                    or nxg._adj[nid][mid] is not nxg._adj[mid][nid]):
                raise ValueError("graph adjacency is not symmetric/aliased")
        adjacency.append([mid, neighbors])
    edges = []
    for left, right, attrs in nxg.edges(data=True):
        _string(left, "edge.left")
        _string(right, "edge.right")
        low, high = sorted((left, right))
        if type(attrs) is not dict:
            raise ValueError("malformed graph edge attributes")
        edges.append([low, high, _plain(attrs, "edge.attributes")])
    edges.sort(key=lambda row: (row[0], row[1]))
    edge_state = {"graph_attrs": _plain(nxg.graph, "graph.graph"),
                  "nodes_order": node_order, "adjacency_order": adjacency, "edges": edges}
    if type(nxg.graph) is not dict:
        raise ValueError("graph-level attributes must be a dictionary")
    # Frozen old graph.py has NO semantic_store member. Its absence is an
    # explicitly authorized representation of the same empty, reconstructible
    # typed substrate that the candidate loader installs on old pickles. The
    # candidate's default path still REFUSES a missing store, and an old graph
    # carrying any unexpected typed member or unknown field is not accepted.
    store = SemanticStore() if old_untyped else graph.semantic_store
    if type(store) is not SemanticStore or set(vars(store)) != _STORE_FIELDS:
        raise ValueError("semantic store has missing/unknown persisted fields")
    if type(store.facts) is not dict:
        raise ValueError("semantic facts must be a dictionary")
    rebuilt = copy.deepcopy(store)
    rebuilt.rebuild_indices(memories=graph.memories)
    facts = []
    for fid in sorted(store.facts):
        fact = store.facts[fid]
        if type(fact) is not SemanticFact or set(vars(fact)) != set(SemanticFact.__dataclass_fields__):
            raise ValueError("semantic fact has missing/unknown persisted fields")
        if fact.id != fid:
            raise ValueError("semantic fact ID disagrees with key")
        fields = {k: _plain(v, f"semantic_fact.{k}") for k, v in vars(fact).items()}
        facts.append([_string(fid, "semantic_fact.id"), fields])
    def _store_indexes(s):
        return {"labels": _set_index(s.labels_by_id, "semantic.labels"),
                "label_index": _set_index(s.label_index, "semantic.label_index"),
                "by_subject_relation": _tuple_set_index(
                    s.by_subject_relation, "semantic.by_subject_relation")}
    indexes = _store_indexes(store)
    if indexes != _store_indexes(rebuilt):
        raise ValueError("semantic indexes/aliases do not reconstruct from facts")
    typed_state = {"facts": facts, "indexes": indexes}
    tracker = graph.recall_tracker
    if type(tracker) is not RecallTracker or set(vars(tracker)) != _TRACKER_FIELDS:
        raise ValueError("recall tracker has missing/unknown persisted fields")
    if type(tracker.history) is not list:
        raise ValueError("recall history is not a list")
    for entry in tracker.history:
        if (type(entry) is not dict or set(entry) != {"query", "node_ids", "count"}
                or type(entry["node_ids"]) is not list
                or type(entry["count"]) is not int):
            raise ValueError("malformed recall tracker history")
        _string(entry["query"], "recall.query")
        for node in entry["node_ids"]:
            _string(node, "recall.node_ids")
    from collections import defaultdict
    counter = tracker.co_recall_count
    if type(counter) is defaultdict:
        # The frozen learning.py persists exactly defaultdict(float). A
        # different default factory changes missing-key behavior and is
        # unaccounted persisted state, so it fails closed.
        if counter.default_factory is not float:
            raise ValueError(
                "recall tracker counts use an unexpected defaultdict factory")
        container = {"kind": "defaultdict", "factory": "float"}
        counter_pairs = dict(counter)
    elif type(counter) is dict:
        container = {"kind": "dict", "factory": None}
        counter_pairs = counter
    else:
        raise ValueError("recall tracker counts are not a mapping")
    co_recall = {"container": container,
                 "pairs": _pairs(counter_pairs, "recall.co_recall_count")}
    graph_state = {
        "similarity_threshold": _finite(graph.similarity_threshold, "similarity_threshold"),
        "model": _string(graph._model_name, "_model_name"),
        "is_bge": graph._is_bge,
        "embedder_cache": None,  # required absent above, not a silent exclusion
        "bm25_cache": _bm25(graph._bm25),
        "bm25_corpus": corpus,
        "entity_index": entities,
        "learned_weights": _pairs(graph.learned_weights, "learned_weights"),
        "recall_tracker": {"graph_dir": _string(tracker.graph_dir, "tracker.graph_dir"),
                           "history": _plain(tracker.history, "tracker.history"),
                           "co_recall_count": co_recall},
    }
    if type(graph._is_bge) is not bool:
        raise ValueError("_is_bge must be boolean")
    full = {"schema": SCHEMA, "memory_order": memory_order,
            "memories": memories, "graph": edge_state,
            "semantic": typed_state, "learning_and_indexes": graph_state}
    return {"schema": SCHEMA,
            "counts": {"memories": len(memories), "nodes": len(node_order),
                       "edges": len(edges), "typed_facts": len(facts)},
            "sha256": {"memories": _digest([memory_order, memories]),
                       "edges": _digest(edge_state),
                       "typed_facts": _digest(typed_state),
                       "full": _digest(full)}}
