#!/usr/bin/env python3
"""Cold-bootstrap a new-format QMG graph pair from an old-generation pair.

The release's strict validator (quantum_memory_graph.rollback._load_pair_checked)
refuses a pickled MemoryGraph that carries attributes outside _GRAPH_ATTRS.
Old-generation pairs carry three vestigial attributes -- ``sources``,
``_source_index`` and ``max_connections`` -- that the new code never reads and
never writes, so ``MemoryGraph.load`` -> ``save`` keeps them and the pair stays
rejected.  This script rebuilds the object in the new shape, proving first that
nothing in those attributes carries per-memory data.

Usage:
    cold_bootstrap.py SRC_DIR DEST_DIR [--model MODEL] [--force]

Requires PYTHONPATH to point at the release package root.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

from quantum_memory_graph import rollback
from quantum_memory_graph.graph import MemoryGraph
from quantum_memory_graph.semantic import SemanticStore

VESTIGIAL = ("sources", "_source_index", "max_connections")
PAIR_FILES = ("graph.pkl", "graph.json")
RECEIPT_NAME = "bootstrap_receipt.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_file(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _is_empty_container(value) -> bool:
    """True only for an empty mapping / sequence (strings count as data)."""
    if isinstance(value, (str, bytes, bytearray)):
        return False
    try:
        return len(value) == 0
    except TypeError:
        return False


def _contained_container(value) -> bool:
    return hasattr(value, "__len__") and not isinstance(
        value, (str, bytes, bytearray))


# --------------------------------------------------------------------------- #
# step 1: source validation
# --------------------------------------------------------------------------- #

def _validate_source(src_dir: str) -> str:
    src = os.path.abspath(os.fspath(src_dir))
    if os.path.islink(src):
        raise ValueError(f"source {src!r} is a symlink; refusing")
    if not os.path.isdir(src):
        raise ValueError(f"source {src!r} is not a directory")
    for name in PAIR_FILES:
        path = os.path.join(src, name)
        if os.path.islink(path):
            raise ValueError(f"source {name!r} is a symlink; refusing")
        if not os.path.isfile(path):
            raise ValueError(f"source is missing required file {name!r}")
    return src


# --------------------------------------------------------------------------- #
# step 3: data-loss guard
# --------------------------------------------------------------------------- #

def _guard_vestigial(graph: MemoryGraph) -> dict:
    """Prove the three dropped attributes carry no per-memory data."""
    for name in ("sources", "_source_index"):
        if not hasattr(graph, name):
            continue
        value = getattr(graph, name)
        if value is None:
            continue
        if not _contained_container(value):
            raise ValueError(
                f"refusing: {name} is {type(value).__name__}, not a mapping/"
                "sequence; cannot prove it is empty")
        if not _is_empty_container(value):
            raise ValueError(
                f"refusing: non-empty {name} map would lose data "
                f"(len={len(value)})")

    if hasattr(graph, "max_connections"):
        mc = graph.max_connections
        if type(mc) is not int:  # bool is not a plain int here
            raise ValueError(
                "refusing: max_connections is not a plain int "
                f"({type(mc).__name__})")

    extra = sorted(set(vars(graph)) - rollback._GRAPH_ATTRS - set(VESTIGIAL))
    if extra:
        raise ValueError(
            f"refusing: unknown graph attributes outside the allowlist: {extra}")

    return {
        "sources_was": (len(graph.sources) if hasattr(graph, "sources")
                        and _contained_container(graph.sources) else None),
        "source_index_was": (len(graph._source_index)
                             if hasattr(graph, "_source_index")
                             and _contained_container(graph._source_index) else None),
        "max_connections_was": getattr(graph, "max_connections", None),
    }


# --------------------------------------------------------------------------- #
# step 4: rebuild in the new shape
# --------------------------------------------------------------------------- #

def _rebuild(graph: MemoryGraph, model: str | None) -> MemoryGraph:
    new = MemoryGraph(
        similarity_threshold=graph.similarity_threshold,
        model=model or graph._model_name,
    )
    new.graph = graph.graph                       # nx.Graph, nodes carry 'memory'
    new.memories = graph.memories
    new.recall_tracker = graph.recall_tracker
    new.learned_weights = dict(getattr(graph, "learned_weights", {}) or {})
    store = getattr(graph, "semantic_store", None)
    new.semantic_store = store if store is not None else SemanticStore()
    new._embedder = None        # unverifiable pickled model cache
    new._bm25 = None
    new._bm25_corpus = []
    new._rebuild_indices()      # _entity_index + _bm25_corpus from memories

    if set(vars(new)) != set(rollback._GRAPH_ATTRS):
        missing = sorted(set(rollback._GRAPH_ATTRS) - set(vars(new)))
        extra = sorted(set(vars(new)) - set(rollback._GRAPH_ATTRS))
        raise ValueError(
            f"rebuilt graph does not match _GRAPH_ATTRS (missing={missing}, "
            f"extra={extra})")
    return new


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #

def cold_bootstrap_pair(src_dir, dest_dir, *, model=None, force=False) -> dict:
    src = _validate_source(src_dir)
    dest = os.path.abspath(os.fspath(dest_dir))
    rollback._assert_disjoint_destination(dest, src)

    if os.path.lexists(dest) and not force:
        raise ValueError(f"destination {dest!r} already exists; use --force")

    # 2. release's own loader
    graph = MemoryGraph.load(os.path.join(src, "graph.pkl"), model=model)

    # 3. data-loss guard before anything is dropped
    was = _guard_vestigial(graph)

    src_memories = len(graph.memories)
    src_edges = graph.graph.number_of_edges()

    # 4. rebuild
    new = _rebuild(graph, model)

    # 5. counts must survive the rebuild
    if len(new.memories) != src_memories:
        raise ValueError("rebuilt memory count disagrees with source")
    if new.graph.number_of_edges() != src_edges:
        raise ValueError("rebuilt edge count disagrees with source")

    # 6. write to a private staging dir next to the destination
    parent = os.path.dirname(dest) or "."
    os.makedirs(parent, exist_ok=True)
    stage = tempfile.mkdtemp(prefix=".cb-stage-", dir=parent)
    try:
        os.chmod(stage, 0o700)
        pkl_path = os.path.join(stage, "graph.pkl")
        json_path = os.path.join(stage, "graph.json")
        new.save(pkl_path)
        new.save_json(json_path)
        for path in (pkl_path, json_path):
            os.chmod(path, 0o600)
            _fsync_file(path)
        _fsync_dir(stage)

        pkl_sha = _sha256_file(pkl_path)
        json_sha = _sha256_file(json_path)

        # 7. self-check with the release's own validator
        checked, _doc, _pb, _jb, _dig = rollback._load_pair_checked(
            stage, "cold bootstrap output", model=model)
        if len(checked.memories) != src_memories:
            raise ValueError(
                f"validator memory count {len(checked.memories)} != source "
                f"{src_memories}")
        if checked.graph.number_of_edges() != src_edges:
            raise ValueError("validator edge count disagrees with source")

        # 8. publish
        replaced = None
        if os.path.lexists(dest):
            replaced = f"{dest}.pre-bootstrap-{time.strftime('%Y%m%dT%H%M%S')}"
            os.rename(dest, replaced)
        os.replace(stage, dest)
        stage = None
        _fsync_dir(parent)
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)

    # 9. receipt
    receipt = {
        "src": src,
        "dest": dest,
        "memories": src_memories,
        "edges": src_edges,
        "dropped_attrs": list(VESTIGIAL),
        "sources_len_was": was["sources_was"],
        "source_index_len_was": was["source_index_was"],
        "max_connections_was": was["max_connections_was"],
        "model": model,
        "persisted_model": new._model_name,
        "validated": True,
        "pkl_sha256": pkl_sha,
        "json_sha256": json_sha,
        "replaced_existing": replaced,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    receipt_path = os.path.join(dest, RECEIPT_NAME)
    tmp_receipt = receipt_path + ".tmp"
    fd = os.open(tmp_receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                 0o600)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(receipt, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        pass
    os.replace(tmp_receipt, receipt_path)
    os.chmod(receipt_path, 0o600)
    _fsync_dir(dest)

    # 8b. re-validate the published pair (receipt must not disturb it)
    rollback._load_pair_checked(dest, "published cold bootstrap output",
                                model=model)
    published = {
        "graph.pkl": _sha256_file(os.path.join(dest, "graph.pkl")),
        "graph.json": _sha256_file(os.path.join(dest, "graph.json")),
    }
    if published["graph.pkl"] != pkl_sha or published["graph.json"] != json_sha:
        raise ValueError("published pair bytes differ from validated staging pair")

    return receipt


def _cli(argv) -> int:
    args = [a for a in argv[1:]]
    model = None
    force = False
    positional = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--force":
            force = True
        elif a == "--model":
            i += 1
            if i >= len(args):
                print("error: --model requires a value", file=sys.stderr)
                return 2
            model = args[i]
        elif a.startswith("--model="):
            model = a.split("=", 1)[1]
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        elif a.startswith("-"):
            print(f"error: unknown option {a!r}", file=sys.stderr)
            return 2
        else:
            positional.append(a)
        i += 1

    if len(positional) != 2:
        print("usage: cold_bootstrap.py SRC_DIR DEST_DIR [--model X] [--force]",
              file=sys.stderr)
        return 2

    try:
        receipt = cold_bootstrap_pair(positional[0], positional[1],
                                      model=model, force=force)
    except Exception as exc:  # refusal path: non-zero, no traceback noise
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2),
              file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
