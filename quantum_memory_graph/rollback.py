"""QMG L3 B4 downgrade/rollback: private, verified capture/replay primitives.

Public surface
--------------
``capture_pair(source_dir, dest_dir)``
    Private, verified snapshot of an untyped ``graph.pkl``/``graph.json`` pair
    plus a completion marker, written LAST inside a 0700 staging directory and
    atomically renamed into place.

``prepare_downgrade(baseline_dir, canary_dir, dest_dir)``
    Private downgrade bundle: an old-code-compatible (typed state stripped)
    ``graph.pkl``/``graph.json`` pair, a canonical ``typed-facts.json`` replay
    sidecar, and a ``bundle.json`` manifest with version, sha256 digests,
    inventory, counts and the full canary memory inventory.

``restore_typed(bundle_dir, legacy_dir, dest_dir, *, quarantine_dir=None)``
    Fail-closed re-import of the typed state into the old-code-era pair,
    producing a NEW private, verified pair.

Trust model
-----------
* Inputs are trusted, PRIVATE on-disk pairs produced by this module or by the
  sandbox package. This module never accepts untrusted uploads and makes no
  attempt to sandbox deserialization: unpickling a hostile file is out of
  scope and the caller must only pass files it already trusts.
* The sha256 digests recorded in a bundle manifest or completion marker are an
  accident-corruption check, NOT authentication. A caller that needs
  authenticity must compare the recorded digests against an independently
  recorded value (e.g. one captured before the downgrade).
* All three operations are fail-closed: nothing is published until it has been
  written, re-read and verified inside a private staging directory; a
  completion marker (a file whose name contains ``complete``) is written LAST
  and only then does an atomic, kernel-enforced no-replace rename
  (``renameat2`` with ``RENAME_NOREPLACE``; fails closed when unavailable)
  publish the result. The source pairs are opened read-only and are never
  modified, on success or on failure.
* Corrupt pickles raise (``MemoryGraph.load`` fails closed); JSON is never used
  as a fallback for the pickle.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import logging
import os
import pickle
import re
import secrets
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .graph import Memory, MemoryGraph
from .semantic import (
    RELATION_TYPES,
    SemanticFact,
    SemanticStore,
    _derive_fact_id,
    _entity_type,
    _iso,
    _utc,
    _validate_canonical_fact,
)

__all__ = ["capture_pair", "prepare_downgrade", "restore_typed"]

_MARKER_NAME = ".complete.json"
_TYPED_FORMAT = "qmg-l3-typed-replay"
_FROZEN_LEGACY_SHA256 = "b5a2cae75da002578e74419679d04f351d24a2ebbd7ec8897fd29918bfec646a"
_PAIR_FILES = ("graph.pkl", "graph.json")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_FACT_DOC_KEYS = frozenset({
    "id", "subject_id", "subject_label", "relation_type", "value", "datatype",
    "source", "source_memory_id", "evidence_memory_ids", "provenance",
    "observed_at", "valid_from", "valid_to", "confidence",
})
_TYPED_TOP_KEYS = frozenset({"format", "version", "facts", "labels_by_id"})
_BUNDLE_KEYS = frozenset({
    "version", "sha256", "inventory", "counts", "sources", "memories",
    "baseline_pair_sha256", "canary_pair_sha256",
})
_BUNDLE_DIGEST_NAMES = frozenset({"graph.pkl", "graph.json", "typed-facts.json"})
_MARKER_KEYS = frozenset({"version", "sha256", "counts"})
_MARKER_DIGEST_NAMES = frozenset(
    {"graph.pkl", "graph.json", "typed-facts.json", "bundle.json"}
)
_COUNT_KEYS = frozenset({"memories", "facts"})
_RECORD_KEYS = frozenset({"id", "source", "text_sha256"})
_PAIR_SHA_KEYS = frozenset({"graph.pkl", "graph.json"})
_LEGACY_JSON_KEYS = frozenset({"node_count", "edge_count", "exported_at", "memories"})
_TYPED_JSON_KEYS = _LEGACY_JSON_KEYS | frozenset({"semantic_schema_version", "semantic_facts"})
_MEMORY_JSON_KEYS = frozenset({"id", "text", "source"})
# A future attribute may contain a typed class old code cannot import. Reject
# unknown persisted fields rather than silently carrying or discarding them.
_GRAPH_ATTRS = frozenset({
    "graph", "memories", "similarity_threshold", "_model_name", "_embedder",
    "_is_bge", "_bm25", "_bm25_corpus", "_entity_index",
    "semantic_store", "recall_tracker", "learned_weights",
})
_MEMORY_ATTRS = frozenset(Memory.__dataclass_fields__)


class _IdentityError(ValueError):
    """Fact identity is missing, malformed or non-canonical (ambiguous)."""


class _TimeError(ValueError):
    """A persisted time field is missing, naive or non-canonical (ambiguous)."""


# --------------------------------------------------------------------------- #
# Small primitives
# --------------------------------------------------------------------------- #

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text_sha256(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, indent=2).encode("utf-8")


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_json(data: bytes, label: str) -> Any:
    """Parse JSON with duplicate-key and non-finite-number rejection."""
    def _no_duplicates(pairs):
        out: Dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            out[key] = value
        return out

    def _reject_constant(text: str):
        raise ValueError(f"{label}: non-finite JSON number {text!r} is not allowed")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: is not valid UTF-8") from exc
    try:
        return json.loads(text, object_pairs_hook=_no_duplicates,
                          parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: is not valid JSON") from exc


def _read_regular_file(path: str, label: str) -> bytes:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(f"{label}: symlinked artifacts are not accepted")
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"{label}: artifact is not a regular file")
    # lstat alone is racy: an input can be swapped for a symlink before open.
    # O_NOFOLLOW binds the read to a regular inode; fstat detects replacement.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as fh:
        opened = os.fstat(fh.fileno())
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (st.st_dev, st.st_ino)):
            raise ValueError(f"{label}: artifact changed while opening")
        return fh.read()


def _read_pair_bytes(dirpath: str, label: str) -> Dict[str, bytes]:
    try:
        return {name: _read_regular_file(os.path.join(dirpath, name),
                                         f"{label}/{name}")
                for name in _PAIR_FILES}
    except FileNotFoundError as exc:
        raise ValueError(f"{label}: a required pair file is missing") from exc


def _pair_digests(pair_bytes: Dict[str, bytes]) -> Dict[str, str]:
    return {name: _sha256_bytes(data) for name, data in pair_bytes.items()}


def _find_complete_markers(dirpath: str) -> List[str]:
    found: List[str] = []
    for root, _dirs, files in os.walk(dirpath):
        for name in files:
            if "complete" in name.casefold():
                found.append(os.path.join(root, name))
    return sorted(found)


def _write_private(path: str, data: bytes) -> None:
    with open(path, "xb") as fh:
        os.chmod(path, 0o600)
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# Linux renameat2(2): fail with EEXIST instead of replacing newpath.
_RENAME_NOREPLACE = 0x1
_AT_FDCWD = -100


def _load_renameat2():
    """Return libc's renameat2, or ``None`` when it is unavailable.

    ``use_errno=True`` lets ``ctypes.get_errno()`` report the C errno of the
    syscall so a claimed destination (EEXIST) can be told apart from an
    unsupported operation (ENOSYS/EINVAL). Loaded per call so neither a
    failed load nor a test double is ever cached.
    """
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except (OSError, TypeError):
        return None
    fn = getattr(libc, "renameat2", None)
    if fn is None:
        return None
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                   ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    return fn


def _rename_noreplace(stage: str, dest: str) -> None:
    """Atomically publish ``stage`` as ``dest`` without replacing anything.

    Kernel-enforced with ``renameat2(..., RENAME_NOREPLACE)``: a destination
    claimed between any earlier existence check and this call raises
    ``ValueError`` instead of being silently replaced. When the symbol or the
    flag is unavailable the publication fails closed; ``os.rename`` and
    ``os.replace`` are never used as fallbacks.
    """
    fn = _load_renameat2()
    if fn is None:
        raise OSError(errno.ENOSYS,
                      "renameat2(RENAME_NOREPLACE) is unavailable; "
                      "refusing non-atomic publication", dest)
    rc = fn(_AT_FDCWD, os.fsencode(stage), _AT_FDCWD, os.fsencode(dest),
            _RENAME_NOREPLACE)
    if rc == 0:
        return
    err = ctypes.get_errno()
    if err in (errno.EEXIST, errno.ENOTEMPTY):
        raise ValueError("destination appeared while publishing; refusing overwrite")
    if err in (errno.ENOSYS, errno.EINVAL):
        raise OSError(err,
                      "renameat2(RENAME_NOREPLACE) is unavailable or "
                      "unsupported; refusing non-atomic publication", dest)
    raise OSError(err or errno.EIO, os.strerror(err or errno.EIO), dest)


def _publish_stage(stage: str, dest: str) -> None:
    """Make the completed generation durable before/after atomic publication.

    The preliminary ``lexists`` is only a fast rejection; the no-replace
    rename below is the authority against a destination claimed in between.
    """
    _fsync_dir(stage)
    if os.path.lexists(dest):
        raise ValueError("destination appeared while publishing; refusing overwrite")
    _rename_noreplace(stage, dest)
    try:
        _fsync_dir(os.path.dirname(os.path.abspath(dest)))
    except BaseException:
        # A reported failure cannot leave an apparently completed destination.
        shutil.rmtree(dest, ignore_errors=True)
        raise


def _make_stage(dest: str) -> str:
    parent = os.path.dirname(os.path.abspath(dest))
    os.makedirs(parent, exist_ok=True)
    name = "." + (os.path.basename(dest) or "output") + ".qmg-stage-" + secrets.token_hex(8)
    stage = os.path.join(parent, name)
    os.mkdir(stage, 0o700)
    return stage


def _cleanup_stage(stage: str) -> None:
    shutil.rmtree(stage, ignore_errors=True)


def _prepare_destination(dest: str) -> None:
    """Never delete an existing path, even if it has no completion marker."""
    if os.path.lexists(dest):
        raise ValueError("destination already exists; use a fresh private path")


def _assert_disjoint_destination(dest: str, *inputs: str) -> None:
    """An output may neither replace nor contain/be contained by an input."""
    target = os.path.realpath(os.path.abspath(dest))
    for source in inputs:
        original = os.path.realpath(os.path.abspath(source))
        shared = os.path.commonpath((target, original))
        if shared == target or shared == original:
            raise ValueError("destination overlaps an input; refusing data loss")


# --------------------------------------------------------------------------- #
# Pair loading / validation
# --------------------------------------------------------------------------- #

def _load_pair_checked(dirpath: str, label: str, *, model: str = None):
    """Load and fully validate a trusted private graph pair.

    Returns ``(graph, json_doc, pkl_bytes, json_bytes, digests)``. The pickle is
    loaded via ``MemoryGraph.load`` (fails closed on corruption; JSON is never
    a fallback) and ``graph.json`` must agree with it on counts, IDs, text,
    sources and typed state. An explicit model uses the service's QMG_MODEL;
    callers without one keep the B4 rollback helper's original behavior.
    """
    d = os.fspath(dirpath)
    if os.path.islink(d) or not os.path.isdir(d):
        raise ValueError(f"{label}: not a readable directory")
    pair_bytes = _read_pair_bytes(d, label)
    digests = _pair_digests(pair_bytes)
    # MemoryGraph.load upgrades an old dict into a new object. Inspect the raw
    # container first so a future typed key cannot be silently discarded.
    original = pickle.loads(pair_bytes["graph.pkl"])
    if isinstance(original, dict):
        if set(original) != {"G", "memories"}:
            raise ValueError(f"{label}: unknown legacy pickle keys; refusing data loss")
    elif isinstance(original, MemoryGraph):
        if set(vars(original)) - _GRAPH_ATTRS:
            raise ValueError(f"{label}: unknown graph attributes; refusing downgrade")
        if model is not None and original._model_name != model:
            # MemoryGraph.load(model=...) rewrites this persisted identifier;
            # a mismatched model cannot be vouched for by a v2 pair receipt.
            raise ValueError(f"{label}: persisted model does not match unit model")
        if model and original._embedder is not None:
            # Legacy pair from a release that pickled the encoder cache. MemoryGraph.load clears it
            # anyway (the cache rebuilds lazily from _model_name), so clean it instead of refusing:
            # an old pair must stay restartable. The model-match check above still guarantees we
            # never silently rebind a different encoder.
            logging.getLogger(__name__).warning(
                "%s: discarding persisted embedder cache (legacy pair); it rebuilds lazily", label)
            original._embedder = None
    else:
        raise ValueError(f"{label}: unsupported pickle format")
    graph = MemoryGraph.load(os.path.join(d, "graph.pkl"), model=model)
    if not isinstance(graph, MemoryGraph):
        raise ValueError(f"{label}: unsupported pickle format")
    if set(vars(graph)) - _GRAPH_ATTRS:
        raise ValueError(f"{label}: unknown graph attributes; refusing downgrade")
    if _pair_digests(_read_pair_bytes(d, label)) != digests:
        raise ValueError(f"{label}: changed while it was being read")
    doc = _strict_json(pair_bytes["graph.json"], f"{label}/graph.json")
    _validate_pair_consistency(graph, doc, label)
    return graph, doc, pair_bytes["graph.pkl"], pair_bytes["graph.json"], digests


def _validate_pair_consistency(graph: MemoryGraph, doc: Any, label: str) -> None:
    if not isinstance(doc, dict) or set(doc) not in (_LEGACY_JSON_KEYS, _TYPED_JSON_KEYS):
        raise ValueError(f"{label}: graph.json has missing or unknown schema fields")
    if (type(doc["node_count"]) is not int or doc["node_count"] < 0
            or type(doc["edge_count"]) is not int or doc["edge_count"] < 0
            or not isinstance(doc["exported_at"], str) or not doc["exported_at"].strip()):
        raise ValueError(f"{label}: graph.json counts or export stamp are malformed")
    memories = graph.memories
    if not isinstance(memories, dict) or set(graph.graph.nodes) != set(memories):
        raise ValueError(f"{label}: graph nodes disagree with memory IDs")
    raw = doc["memories"]
    if not isinstance(raw, list):
        raise ValueError(f"{label}: graph.json is missing its memory list")
    by_id: Dict[str, Any] = {}
    for entry in raw:
        if (not isinstance(entry, dict) or set(entry) != _MEMORY_JSON_KEYS
                or not isinstance(entry["id"], str) or not entry["id"]
                or not isinstance(entry["text"], str)
                or not isinstance(entry["source"], str)):
            raise ValueError(f"{label}: graph.json has malformed or unknown memory fields")
        if entry["id"] in by_id:
            raise ValueError(f"{label}: graph.json repeats a memory ID")
        by_id[entry["id"]] = entry
    if set(by_id) != set(memories):
        raise ValueError(f"{label}: graph.json memory IDs disagree with the pickle")
    if doc["node_count"] != len(memories):
        raise ValueError(f"{label}: graph.json node_count disagrees with the pickle")
    if doc["edge_count"] != graph.graph.number_of_edges():
        raise ValueError(f"{label}: graph.json edge_count disagrees with the pickle")
    for mid, mem in memories.items():
        if (not isinstance(mid, str) or not mid or not isinstance(mem, Memory)
                or set(vars(mem)) - _MEMORY_ATTRS or mem.id != mid
                or not isinstance(mem.text, str)
                or not isinstance(mem.source, str)):
            raise ValueError(f"{label}: pickle memory has an invalid ID or unknown fields")
        entry = by_id[mid]
        if entry["text"] != mem.text or entry["source"] != mem.source:
            raise ValueError(f"{label}: graph.json text/source disagrees with the pickle")
        node = graph.graph.nodes[mid]
        if set(node) - {"memory"}:
            raise ValueError(f"{label}: graph node has unknown fields")
        linked = node.get("memory")
        if linked is not None and (
                not isinstance(linked, Memory) or linked.id != mid
                or linked.text != mem.text or linked.source != mem.source
                or set(vars(linked)) - _MEMORY_ATTRS):
            raise ValueError(f"{label}: graph node memory disagrees with the pickle")
    for _u, _v, edge in graph.graph.edges(data=True):
        if set(edge) - {"weight", "types"}:
            raise ValueError(f"{label}: graph edge has unknown fields")
        if "types" in edge and (not isinstance(edge["types"], list)
                or not all(isinstance(kind, str) for kind in edge["types"])):
            raise ValueError(f"{label}: graph edge types are malformed")
    store = getattr(graph, "semantic_store", None)
    facts = getattr(store, "facts", None) if store is not None else None
    if set(doc) == _TYPED_JSON_KEYS:
        if (type(doc["semantic_schema_version"]) is not int
                or doc["semantic_schema_version"] != 1
                or not isinstance(doc["semantic_facts"], list)):
            raise ValueError(f"{label}: unsupported or malformed typed JSON schema")
        expected = [facts[fid].to_dict() for fid in sorted(facts)] if facts else []
        if _canonical_json_bytes(doc["semantic_facts"]) != _canonical_json_bytes(expected):
            raise ValueError(f"{label}: graph.json typed facts disagree with the pickle")
    elif facts:
        raise ValueError(f"{label}: typed facts present without a typed JSON schema")


# --------------------------------------------------------------------------- #
# capture_pair
# --------------------------------------------------------------------------- #

def capture_pair(source_dir, dest_dir) -> None:
    """Create a private, verified copy of a graph pair plus a completion marker.

    Rejects symlinked or non-regular artifacts, corrupt pickles and any
    pickle/JSON inconsistency without producing output; the source pair is
    hashed before and after the copy and compared against the private copies.
    """
    src = os.fspath(source_dir)
    dest = os.fspath(dest_dir)
    if os.path.islink(dest) or os.path.islink(src):
        raise ValueError("pair artifacts must not be symlinks")
    _assert_disjoint_destination(dest, src)
    _prepare_destination(dest)

    graph, _doc, pkl_bytes, json_bytes, digests = _load_pair_checked(src, "source pair")
    if _pair_digests(_read_pair_bytes(src, "source pair")) != digests:
        raise ValueError("source pair changed while it was being read")

    stage = _make_stage(dest)
    try:
        _write_private(os.path.join(stage, "graph.pkl"), pkl_bytes)
        _write_private(os.path.join(stage, "graph.json"), json_bytes)
        copied = _pair_digests(_read_pair_bytes(stage, "staged capture"))
        if copied != digests:
            raise ValueError("staged capture does not reproduce the source pair")
        if _pair_digests(_read_pair_bytes(src, "source pair")) != digests:
            raise ValueError("source pair changed during the capture")
        reloaded = MemoryGraph.load(os.path.join(stage, "graph.pkl"))
        if set(reloaded.memories) != set(graph.memories):
            raise ValueError("staged capture is not loadable as the source graph")
        marker = {
            "version": 1,
            "sha256": copied,
            "counts": {"memories": len(graph.memories)},
        }
        _write_private(os.path.join(stage, _MARKER_NAME), _json_bytes(marker))
        os.chmod(stage, 0o700)
        _publish_stage(stage, dest)
    except BaseException:
        _cleanup_stage(stage)
        raise


# --------------------------------------------------------------------------- #
# prepare_downgrade
# --------------------------------------------------------------------------- #

def _strip_typed_state(graph: MemoryGraph) -> MemoryGraph:
    """Copy the graph with ONLY typed state removed for frozen old code.

    The embedder must already have a verified detach receipt. BM25 is included
    in inventory v2 and can affect search ranking; silently dropping it would
    invalidate the no-loss old-code handoff.
    """
    if graph._embedder is not None:
        raise ValueError("downgrade requires a verified detached embedder cache")
    stripped = MemoryGraph.__new__(MemoryGraph)
    stripped.__dict__.update(graph.__dict__)
    stripped.__dict__.pop("semantic_store", None)
    return stripped


def _require_untyped_baseline(graph: MemoryGraph) -> None:
    store = getattr(graph, "semantic_store", None)
    if store is not None and getattr(store, "facts", None):
        raise ValueError("baseline pair carries typed facts; refusing a typed baseline")


def _require_baseline_subset(baseline: MemoryGraph, canary: MemoryGraph) -> None:
    for mid, mem in baseline.memories.items():
        other = canary.memories.get(mid)
        if other is None:
            raise ValueError("canary pair dropped a baseline memory; refusing deletions")
        if other.text != mem.text or getattr(other, "source", "") != getattr(mem, "source", ""):
            raise ValueError("canary pair altered a baseline memory; refusing deletions")


def _verify_downgraded_bundle(stage: str, canary: MemoryGraph) -> None:
    staged = MemoryGraph.load(os.path.join(stage, "graph.pkl"))
    if getattr(staged.semantic_store, "facts", None):
        raise ValueError("downgraded pickle still carries typed state")
    if set(staged.memories) != set(canary.memories):
        raise ValueError("downgraded pickle lost or added memories")
    for mid, mem in canary.memories.items():
        got = staged.memories[mid]
        if got.text != mem.text or getattr(got, "source", "") != getattr(mem, "source", ""):
            raise ValueError("downgraded pickle altered a memory")
    if set(staged.graph.nodes()) != set(staged.memories):
        raise ValueError("downgraded pickle lost a graph node")
    for u, v in canary.graph.edges():
        if not staged.graph.has_edge(u, v):
            raise ValueError("downgraded pickle lost a graph edge")
    if canary.graph.number_of_edges() != staged.graph.number_of_edges():
        raise ValueError("downgraded pickle edge set is inconsistent")
    doc = _strict_json(
        _read_regular_file(os.path.join(stage, "graph.json"), "staged graph.json"),
        "staged graph.json",
    )
    if not isinstance(doc, dict) or set(doc) != {
        "node_count", "edge_count", "exported_at", "memories"
    }:
        raise ValueError("downgraded graph.json is not legacy-shaped")
    _validate_pair_consistency(staged, doc, "staged downgrade")


def _verify_frozen_old_loader(stage: str, canary: MemoryGraph) -> None:
    """Prove that the *pinned old interpreter*, not just new code, loads it.

    A future-only object hidden inside an otherwise allowed legacy memory
    field can survive removal of ``semantic_store``. The new loader would
    accept it while real old code fails at unpickle time. Probe the actual
    complete staged pair using the frozen old package in a separate process.
    """
    legacy_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "legacy-code")
    if os.path.islink(legacy_root) or not os.path.isdir(legacy_root):
        raise ValueError("frozen old-code package is required for downgrade")
    sidecar = _read_regular_file(os.path.join(legacy_root, "LEGACY_CODE_SHA256SUMS.txt"),
                                 "frozen old-code SHA sidecar")
    if _sha256_bytes(sidecar) != _FROZEN_LEGACY_SHA256:
        raise ValueError("frozen old-code SHA sidecar changed")
    recorded: Dict[str, str] = {}
    for line in sidecar.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if (len(parts) != 4 or parts[0] != parts[1]
                or not _HEX64.fullmatch(parts[0]) or parts[3] != "OK"
                or not parts[2].endswith(".py") or "/" in parts[2]
                or parts[2] in recorded):
            raise ValueError("frozen old-code file receipt is malformed")
        recorded[parts[2]] = parts[0]
    package_dir = os.path.join(legacy_root, "quantum_memory_graph")
    actual = {name for name in os.listdir(package_dir) if name.endswith(".py")}
    if actual != set(recorded) or not actual:
        raise ValueError("frozen old-code file inventory changed")
    for name, expected in recorded.items():
        if _sha256_bytes(_read_regular_file(os.path.join(package_dir, name),
                                             "frozen old-code module")) != expected:
            raise ValueError("frozen old-code module bytes changed")

    probe = r'''
import hashlib, json, os, sys
import quantum_memory_graph
from quantum_memory_graph.graph import MemoryGraph
root = os.path.realpath(sys.argv[1])
origin = os.path.realpath(quantum_memory_graph.__file__)
if origin != os.path.join(root, "quantum_memory_graph", "__init__.py"):
    raise SystemExit("old-code import origin mismatch")
graph = MemoryGraph.load(sys.argv[2])
records = sorted(({
    "id": mid,
    "source_sha256": hashlib.sha256(mem.source.encode("utf-8")).hexdigest(),
    "text_sha256": hashlib.sha256(mem.text.encode("utf-8")).hexdigest(),
} for mid, mem in graph.memories.items()), key=lambda row: row["id"])
print(json.dumps({"origin": origin, "nodes": graph.graph.number_of_nodes(),
                  "edges": graph.graph.number_of_edges(), "memories": records}))
'''
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/home/dt"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONPATH": legacy_root,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-c", probe, legacy_root,
             os.path.join(stage, "graph.pkl")],
            cwd=stage, env=environment, capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("frozen old-code compatibility probe could not finish") from exc
    if result.returncode:
        raise ValueError("staged downgrade is not readable by frozen old-code loader")
    try:
        old = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise ValueError("old-code loader did not report its inventory") from exc
    expected_records = sorted(({
        "id": mid,
        "source_sha256": _sha256_bytes(mem.source.encode("utf-8")),
        "text_sha256": _text_sha256(mem.text),
    } for mid, mem in canary.memories.items()), key=lambda row: row["id"])
    if (set(old) != {"origin", "nodes", "edges", "memories"}
            or old["origin"] != os.path.realpath(os.path.join(package_dir, "__init__.py"))
            or type(old["nodes"]) is not int or old["nodes"] != canary.graph.number_of_nodes()
            or type(old["edges"]) is not int or old["edges"] != canary.graph.number_of_edges()
            or old["memories"] != expected_records):
        raise ValueError("frozen old-code loader lost a downgraded memory or source")


def prepare_downgrade(baseline_dir, canary_dir, dest_dir) -> None:
    """Write a private downgrade bundle: old-code pair + typed replay sidecar.

    Both input pairs are loaded, fully cross-validated and required to be
    stable while read. The baseline must be untyped and must persist entirely
    (ID/text/source) in the canary.
    """
    base = os.fspath(baseline_dir)
    can = os.fspath(canary_dir)
    dest = os.fspath(dest_dir)
    if os.path.islink(dest) or os.path.islink(base) or os.path.islink(can):
        raise ValueError("pair artifacts must not be symlinks")
    _assert_disjoint_destination(dest, base, can)
    _prepare_destination(dest)

    base_graph, _base_doc, _bp, _bj, base_digests = _load_pair_checked(base, "baseline pair")
    can_graph, _can_doc, _cp, _cj, can_digests = _load_pair_checked(can, "canary pair")
    _require_untyped_baseline(base_graph)
    _require_baseline_subset(base_graph, can_graph)
    if (_pair_digests(_read_pair_bytes(base, "baseline pair")) != base_digests
            or _pair_digests(_read_pair_bytes(can, "canary pair")) != can_digests):
        raise ValueError("input pair changed while it was being read")

    facts = [can_graph.semantic_store.facts[fid].to_dict()
             for fid in sorted(can_graph.semantic_store.facts)]
    labels = {sid: sorted(labels)
              for sid, labels in can_graph.semantic_store.labels_by_id.items()}
    typed_doc = {
        "format": _TYPED_FORMAT,
        "version": 1,
        "facts": facts,
        "labels_by_id": labels,
    }
    typed_bytes = _canonical_json_bytes(typed_doc)

    stripped = _strip_typed_state(can_graph)
    pkl_bytes = pickle.dumps(stripped)
    legacy_doc = {
        "node_count": stripped.graph.number_of_nodes(),
        "edge_count": stripped.graph.number_of_edges(),
        "exported_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        "memories": [
            {"id": mid, "text": mem.text, "source": getattr(mem, "source", "")}
            for mid, mem in stripped.memories.items()
        ],
    }
    json_bytes = _json_bytes(legacy_doc)
    records = sorted(
        ({"id": mid,
          "source": getattr(mem, "source", ""),
          "text_sha256": _text_sha256(mem.text)}
         for mid, mem in can_graph.memories.items()),
        key=lambda record: record["id"],
    )

    stage = _make_stage(dest)
    try:
        _write_private(os.path.join(stage, "graph.pkl"), pkl_bytes)
        _write_private(os.path.join(stage, "graph.json"), json_bytes)
        _write_private(os.path.join(stage, "typed-facts.json"), typed_bytes)
        digests = {
            "graph.pkl": _sha256_bytes(pkl_bytes),
            "graph.json": _sha256_bytes(json_bytes),
            "typed-facts.json": _sha256_bytes(typed_bytes),
        }
        bundle_doc = {
            "version": 1,
            "sha256": {
                "graph.pkl": digests["graph.pkl"],
                "graph.json": digests["graph.json"],
                "typed-facts.json": digests["typed-facts.json"],
            },
            "inventory": ["graph.json", "graph.pkl", "typed-facts.json"],
            "counts": {"memories": len(records), "facts": len(facts)},
            "sources": sorted({record["source"] for record in records}),
            "memories": records,
            "baseline_pair_sha256": {
                "graph.pkl": base_digests["graph.pkl"],
                "graph.json": base_digests["graph.json"],
            },
            "canary_pair_sha256": {
                "graph.pkl": can_digests["graph.pkl"],
                "graph.json": can_digests["graph.json"],
            },
        }
        bundle_bytes = _json_bytes(bundle_doc)
        _write_private(os.path.join(stage, "bundle.json"), bundle_bytes)
        _verify_downgraded_bundle(stage, can_graph)
        _verify_frozen_old_loader(stage, can_graph)
        marker = {
            "version": 1,
            "sha256": {
                "graph.pkl": digests["graph.pkl"],
                "graph.json": digests["graph.json"],
                "typed-facts.json": digests["typed-facts.json"],
                "bundle.json": _sha256_bytes(bundle_bytes),
            },
            "counts": {"memories": len(records), "facts": len(facts)},
        }
        _write_private(os.path.join(stage, _MARKER_NAME), _json_bytes(marker))
        if (_pair_digests(_read_pair_bytes(base, "baseline pair")) != base_digests
                or _pair_digests(_read_pair_bytes(can, "canary pair")) != can_digests):
            raise ValueError("input pair changed during the downgrade")
        os.chmod(stage, 0o700)
        _publish_stage(stage, dest)
    except BaseException:
        _cleanup_stage(stage)
        raise


# --------------------------------------------------------------------------- #
# restore_typed
# --------------------------------------------------------------------------- #

def _read_bundle_file(bundle: str, name: str) -> bytes:
    try:
        return _read_regular_file(os.path.join(bundle, name), f"bundle/{name}")
    except FileNotFoundError as exc:
        raise ValueError(f"bundle/{name}: required file is missing") from exc


def _validate_bundle_manifest(doc: Any, bundle: str) -> None:
    if not isinstance(doc, dict) or set(doc) != _BUNDLE_KEYS:
        raise ValueError("bundle.json does not match the expected schema")
    if type(doc["version"]) is not int or doc["version"] != 1:
        raise ValueError("unsupported bundle version")
    digests = doc["sha256"]
    if not isinstance(digests, dict) or set(digests) != _BUNDLE_DIGEST_NAMES:
        raise ValueError("bundle.json sha256 map does not cover exactly the required files")
    for name in sorted(digests):
        digest = digests[name]
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise ValueError(f"bundle.json digest for {name} is not a sha256 hex digest")
        if _sha256_bytes(_read_bundle_file(bundle, name)) != digest:
            raise ValueError(f"bundle digest mismatch for {name}")
    inventory = doc["inventory"]
    if (not isinstance(inventory, list)
            or not all(isinstance(name, str) for name in inventory)
            or len(inventory) != len(set(inventory))
            or set(inventory) != set(_BUNDLE_DIGEST_NAMES)):
        raise ValueError("bundle.json inventory does not match its digest map")
    counts = doc["counts"]
    if (not isinstance(counts, dict) or set(counts) != _COUNT_KEYS
            or any(type(counts[key]) is not int for key in _COUNT_KEYS)):
        raise ValueError("bundle.json counts are malformed")
    sources = doc["sources"]
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        raise ValueError("bundle.json sources are malformed")
    if sources != sorted(set(sources)):
        raise ValueError("bundle.json sources must be sorted and unique")
    records = doc["memories"]
    if not isinstance(records, list):
        raise ValueError("bundle.json memories are malformed")
    seen: set = set()
    ids: List[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
            raise ValueError("bundle.json memory record has unknown or missing fields")
        rid = record["id"]
        if not isinstance(rid, str) or not rid:
            raise ValueError("bundle.json memory record id is malformed")
        if rid in seen:
            raise ValueError("bundle.json repeats a memory id")
        seen.add(rid)
        ids.append(rid)
        if not isinstance(record["source"], str):
            raise ValueError("bundle.json memory record source is malformed")
        if not isinstance(record["text_sha256"], str) or not _HEX64.fullmatch(record["text_sha256"]):
            raise ValueError("bundle.json memory record text digest is malformed")
    if ids != sorted(ids):
        raise ValueError("bundle.json memory records must be sorted by id")
    for key in ("baseline_pair_sha256", "canary_pair_sha256"):
        pair_map = doc[key]
        if not isinstance(pair_map, dict) or set(pair_map) != _PAIR_SHA_KEYS:
            raise ValueError(f"bundle.json {key} is malformed")
        for value in pair_map.values():
            if not isinstance(value, str) or not _HEX64.fullmatch(value):
                raise ValueError(f"bundle.json {key} digest is malformed")


def _validate_bundle_marker(bundle: str, bundle_doc: Dict[str, Any]) -> Dict[str, Any]:
    markers = _find_complete_markers(bundle)
    if not markers:
        raise ValueError("bundle has no completion marker; refusing an incomplete bundle")
    doc: Any = None
    for path in markers:
        doc = _strict_json(_read_regular_file(path, "bundle completion marker"),
                           "bundle completion marker")
        if not isinstance(doc, dict) or set(doc) != _MARKER_KEYS:
            raise ValueError("bundle completion marker does not match the expected schema")
        if type(doc["version"]) is not int or doc["version"] != 1:
            raise ValueError("unsupported completion marker version")
        digests = doc["sha256"]
        if not isinstance(digests, dict) or set(digests) != _MARKER_DIGEST_NAMES:
            raise ValueError("completion marker digest map is malformed")
        for name in sorted(digests):
            digest = digests[name]
            if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
                raise ValueError(f"completion marker digest for {name} is malformed")
            if _sha256_bytes(_read_bundle_file(bundle, name)) != digest:
                raise ValueError(f"completion marker digest mismatch for {name}")
        counts = doc["counts"]
        if (not isinstance(counts, dict) or set(counts) != _COUNT_KEYS
                or any(type(counts[key]) is not int for key in _COUNT_KEYS)):
            raise ValueError("completion marker counts are malformed")
        if counts != bundle_doc["counts"]:
            raise ValueError("completion marker counts disagree with the manifest")
    return doc


def _validate_bundle_inventory(doc: Dict[str, Any], fact_count: int) -> None:
    records = doc["memories"]
    if doc["counts"]["memories"] != len(records):
        raise ValueError("bundle.json counts.memories disagrees with its memory records")
    if doc["counts"]["facts"] != fact_count:
        raise ValueError("bundle.json counts.facts disagrees with typed-facts.json")
    if doc["sources"] != sorted({record["source"] for record in records}):
        raise ValueError("bundle.json sources disagree with its memory records")


def _parse_canonical_time(value: Any, name: str, idx: int, *,
                          required: bool) -> Optional[datetime]:
    if value is None:
        if required:
            raise _TimeError(f"fact #{idx} {name} requires an explicit UTC timestamp")
        return None
    if not isinstance(value, str):
        raise _TimeError(f"fact #{idx} {name} must be a canonical UTC ISO8601 string")
    try:
        parsed = _utc(value, name, required=True)
    except ValueError as exc:
        raise _TimeError(f"fact #{idx} {name} is not an aware canonical UTC timestamp") from exc
    if _iso(parsed) != value:
        raise _TimeError(f"fact #{idx} {name} is not the canonical UTC representation")
    return parsed


def _build_fact(fdoc: Dict[str, Any], idx: int,
                labels: Dict[str, List[str]]) -> Tuple[Dict[str, Any], SemanticFact]:
    subject_id = fdoc["subject_id"]
    if not isinstance(subject_id, str) or not subject_id or subject_id != subject_id.strip():
        raise _IdentityError(f"fact #{idx} has a missing or malformed subject_id")
    relation = fdoc["relation_type"]
    if not isinstance(relation, str) or relation not in RELATION_TYPES:
        raise _IdentityError(f"fact #{idx} relation_type is not allowlisted")
    try:
        _entity_type(subject_id)
    except ValueError as exc:
        raise _IdentityError(f"fact #{idx} subject_id is not an allowlisted entity ID") from exc
    if _entity_type(subject_id) != RELATION_TYPES[relation][0]:
        raise _IdentityError(f"fact #{idx} subject_id does not match its relation_type")

    source = fdoc["source"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"fact #{idx} requires a nonblank source")
    source_memory_id = fdoc["source_memory_id"]
    if not isinstance(source_memory_id, str) or not source_memory_id.strip():
        raise _IdentityError(f"fact #{idx} has a missing or malformed source_memory_id")
    evidence = fdoc["evidence_memory_ids"]
    if not isinstance(evidence, list) or not evidence:
        raise _IdentityError(f"fact #{idx} evidence_memory_ids must be a nonempty list")
    seen: List[str] = []
    for eid in evidence:
        if not isinstance(eid, str) or not eid.strip():
            raise _IdentityError(f"fact #{idx} has a malformed evidence memory id")
        if eid in seen:
            raise _IdentityError(f"fact #{idx} repeats an evidence memory id")
        seen.append(eid)
    if source_memory_id not in seen:
        raise _IdentityError(
            f"fact #{idx} source_memory_id does not appear in its evidence list")

    observed_at = _parse_canonical_time(fdoc["observed_at"], "observed_at", idx, required=True)
    valid_from = _parse_canonical_time(fdoc["valid_from"], "valid_from", idx, required=False)
    valid_to = _parse_canonical_time(fdoc["valid_to"], "valid_to", idx, required=False)

    label = fdoc["subject_label"]
    if label is not None and (
            not isinstance(label, str) or not label.strip() or label != label.strip()):
        raise ValueError(f"fact #{idx} subject_label must be canonical nonblank text")
    fields = _validate_canonical_fact(
        subject_id=subject_id,
        relation_type=relation,
        value=fdoc["value"],
        datatype=fdoc["datatype"],
        source=source,
        subject_label=label,
        provenance=fdoc["provenance"],
        observed_at=observed_at,
        valid_from=valid_from,
        valid_to=valid_to,
        confidence=fdoc["confidence"],
        persisted=False,
    )
    derived = _derive_fact_id(
        fields["subject_id"], fields["relation_type"], fields["value"],
        fields["datatype"], fields["source"], fields["provenance"],
        fields["observed_at"], fields["valid_from"], fields["valid_to"],
        fields["confidence"],
    )
    fid = fdoc["id"]
    if not isinstance(fid, str) or not _HEX64.fullmatch(fid) or fid != derived:
        raise _IdentityError(f"fact #{idx} id is not the canonical derived identity")
    if fields["subject_label"] is not None and fields["subject_label"] not in labels.get(subject_id, []):
        raise ValueError(f"fact #{idx} subject_label is absent from labels_by_id")

    fact = SemanticFact(
        id=fid,
        subject_id=fields["subject_id"],
        relation_type=fields["relation_type"],
        value=fields["value"],
        datatype=fields["datatype"],
        source=fields["source"],
        source_memory_id=source_memory_id,
        observed_at=fields["observed_at"],
        subject_label=fields["subject_label"],
        provenance=fields["provenance"],
        valid_from=fields["valid_from"],
        valid_to=fields["valid_to"],
        confidence=fields["confidence"],
        evidence_memory_ids=tuple(seen),
    )
    if _canonical_json_bytes(fact.to_dict()) != _canonical_json_bytes(fdoc):
        raise ValueError(f"fact #{idx} does not round-trip to its canonical form")
    return fdoc, fact


def _validate_typed_facts(doc: Any) -> List[Tuple[Dict[str, Any], SemanticFact]]:
    if not isinstance(doc, dict) or set(doc) != _TYPED_TOP_KEYS:
        raise ValueError("typed-facts.json does not match the expected top-level schema")
    if doc["format"] != _TYPED_FORMAT:
        raise ValueError("typed-facts.json format string is not recognised")
    if type(doc["version"]) is not int or doc["version"] != 1:
        raise ValueError("unsupported typed-facts.json version")
    raw_facts = doc["facts"]
    if not isinstance(raw_facts, list):
        raise ValueError("typed-facts.json facts must be a list")
    labels_raw = doc["labels_by_id"]
    if not isinstance(labels_raw, dict):
        raise ValueError("typed-facts.json labels_by_id must be a mapping")

    labels: Dict[str, List[str]] = {}
    for subject_id, values in labels_raw.items():
        if not isinstance(subject_id, str):
            raise ValueError("labels_by_id keys must be entity ID strings")
        try:
            _entity_type(subject_id)
        except ValueError as exc:
            raise ValueError("labels_by_id key is not an allowlisted entity ID") from exc
        if not isinstance(values, list) or not values:
            raise ValueError("labels_by_id values must be nonempty label lists")
        seen: List[str] = []
        for label in values:
            if not isinstance(label, str) or not label.strip() or label != label.strip():
                raise ValueError("labels_by_id values must be canonical nonblank text")
            if label in seen:
                raise ValueError("labels_by_id values must be unique")
            seen.append(label)
        if seen != sorted(seen):
            raise ValueError("labels_by_id values must be sorted")
        labels[subject_id] = seen

    out: List[Tuple[Dict[str, Any], SemanticFact]] = []
    ids: List[str] = []
    for idx, fdoc in enumerate(raw_facts):
        if not isinstance(fdoc, dict):
            raise ValueError(f"fact #{idx} must be a mapping")
        unknown = set(fdoc) - _FACT_DOC_KEYS
        if unknown:
            raise ValueError(f"fact #{idx} has unknown fields: {sorted(unknown)}")
        missing = _FACT_DOC_KEYS - set(fdoc)
        if "subject_id" in missing:
            raise _IdentityError(
                f"fact #{idx} has no explicit subject_id (label-only identity is ambiguous)")
        if missing:
            raise ValueError(f"fact #{idx} is missing required fields: {sorted(missing)}")
        out.append(_build_fact(fdoc, idx, labels))
        ids.append(fdoc["id"])
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("typed facts must have unique, ascending-sorted IDs")
    return out


def _require_legacy_inventory(doc: Dict[str, Any], legacy: MemoryGraph) -> None:
    for record in doc["memories"]:
        mem = legacy.memories.get(record["id"])
        if mem is None:
            raise ValueError("legacy pair is missing a canary memory ID; refusing deletions")
        if getattr(mem, "source", "") != record["source"]:
            raise ValueError("legacy pair changed a canary memory source; refusing data loss")
        if _text_sha256(mem.text) != record["text_sha256"]:
            raise ValueError("legacy pair changed a canary memory text; refusing data loss")


def _require_legacy_typed_state_idempotent(
        legacy: MemoryGraph, fact_pairs: List[Tuple[Dict[str, Any], SemanticFact]],
        labels: Dict[str, List[str]]) -> None:
    store = getattr(legacy, "semantic_store", None)
    facts = getattr(store, "facts", None) if store is not None else None
    if not facts:
        return
    current = [facts[fid].to_dict() for fid in sorted(facts)]
    expected = [doc for doc, _fact in fact_pairs]
    if _canonical_json_bytes(current) != _canonical_json_bytes(expected):
        raise ValueError("legacy pair carries unrelated typed state; refusing to merge")
    current_labels = {sid: sorted(values) for sid, values in store.labels_by_id.items()}
    if _canonical_json_bytes(current_labels) != _canonical_json_bytes(labels):
        raise ValueError("legacy pair carries aliases that disagree with the bundle")


def _verify_restored_pair(stage: str, restored: MemoryGraph, store: SemanticStore,
                          typed_doc: Dict[str, Any]) -> None:
    staged = MemoryGraph.load(os.path.join(stage, "graph.pkl"))
    if set(staged.memories) != set(restored.memories):
        raise ValueError("restored pickle lost or added memories")
    for mid, mem in restored.memories.items():
        got = staged.memories[mid]
        if got.text != mem.text or getattr(got, "source", "") != getattr(mem, "source", ""):
            raise ValueError("restored pickle altered a memory")
    if set(staged.semantic_store.facts) != set(store.facts):
        raise ValueError("restored pickle lost or added typed facts")
    for fdoc in typed_doc["facts"]:
        got = staged.semantic_store.facts.get(fdoc["id"])
        if got is None:
            raise ValueError("restored pickle dropped a typed fact")
        if _canonical_json_bytes(got.to_dict()) != _canonical_json_bytes(fdoc):
            raise ValueError("restored pickle altered a typed fact")
        for eid in got.evidence_memory_ids:
            if eid not in staged.memories:
                raise ValueError("restored pickle has orphan typed evidence")
    expected_labels = {sid: set(values) for sid, values in typed_doc["labels_by_id"].items()}
    if staged.semantic_store.labels_by_id != expected_labels:
        raise ValueError("restored pickle altered the alias labels")
    doc = _strict_json(
        _read_regular_file(os.path.join(stage, "graph.json"), "staged restored graph.json"),
        "staged restored graph.json",
    )
    if not isinstance(doc, dict) or doc.get("semantic_schema_version") != 1:
        raise ValueError("restored graph.json is missing its typed schema version")
    if _canonical_json_bytes(doc.get("semantic_facts")) != _canonical_json_bytes(typed_doc["facts"]):
        raise ValueError("restored graph.json typed facts do not replay the bundle")
    _validate_pair_consistency(staged, doc, "staged restore")


def restore_typed(bundle_dir, legacy_dir, dest_dir, *, quarantine_dir=None) -> None:
    """Re-import the bundle's typed state into the old-code-era pair.

    Strictly validates the completion marker, manifest schema and digests, the
    typed replay sidecar (schema, canonical identity, canonical UTC times,
    tuple evidence), and the full canary inventory before staging a NEW private
    verified pair. Ambiguous identity or naive/non-canonical times raise
    ``ValueError`` and, when ``quarantine_dir`` is supplied, preserve a
    byte-identical copy of the invalid bundle there first.
    """
    bundle = os.fspath(bundle_dir)
    legacy_path = os.fspath(legacy_dir)
    dest = os.fspath(dest_dir)
    if os.path.islink(dest) or os.path.islink(bundle) or os.path.islink(legacy_path):
        raise ValueError("pair artifacts must not be symlinks")
    _assert_disjoint_destination(dest, bundle, legacy_path)
    if quarantine_dir is not None:
        _assert_disjoint_destination(os.fspath(quarantine_dir), bundle, legacy_path, dest)
    _prepare_destination(dest)

    bundle_doc = _strict_json(_read_bundle_file(bundle, "bundle.json"), "bundle.json")
    _validate_bundle_manifest(bundle_doc, bundle)

    typed_doc = _strict_json(_read_bundle_file(bundle, "typed-facts.json"), "typed-facts.json")
    try:
        fact_pairs = _validate_typed_facts(typed_doc)
    except (_IdentityError, _TimeError) as exc:
        if quarantine_dir is not None:
            _quarantine_bundle(bundle, os.fspath(quarantine_dir), str(exc))
        raise

    marker_doc = _validate_bundle_marker(bundle, bundle_doc)
    _validate_bundle_inventory(bundle_doc, len(fact_pairs))

    legacy_graph, _legacy_doc, _lp, _lj, legacy_digests = _load_pair_checked(
        legacy_path, "legacy pair")
    _require_legacy_inventory(bundle_doc, legacy_graph)
    _require_legacy_typed_state_idempotent(legacy_graph, fact_pairs,
                                           typed_doc["labels_by_id"])

    store = SemanticStore()
    store.facts = {fact.id: fact for _fdoc, fact in fact_pairs}
    store.labels_by_id = {sid: set(values)
                          for sid, values in typed_doc["labels_by_id"].items()}
    store.rebuild_indices(memories=legacy_graph.memories)
    for fdoc, fact in fact_pairs:
        got = store.facts[fdoc["id"]]
        if _canonical_json_bytes(got.to_dict()) != _canonical_json_bytes(fdoc):
            raise ValueError("restored fact does not round-trip to its canonical form")
        for eid in got.evidence_memory_ids:
            if eid not in legacy_graph.memories:
                raise ValueError("restored fact links to a memory outside the legacy pair")
    expected_labels = {sid: set(values) for sid, values in typed_doc["labels_by_id"].items()}
    if store.labels_by_id != expected_labels:
        raise ValueError("restored aliases do not match the bundle")
    legacy_graph.semantic_store = store
    if legacy_graph._embedder is not None:
        raise ValueError("legacy pair requires a verified detached embedder cache")
    # Preserve the v2-covered BM25 corpus AND its materialized ranking cache.

    restored_doc = {
        "node_count": legacy_graph.graph.number_of_nodes(),
        "edge_count": legacy_graph.graph.number_of_edges(),
        "exported_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        "memories": [
            {"id": mid, "text": mem.text, "source": getattr(mem, "source", "")}
            for mid, mem in legacy_graph.memories.items()
        ],
        "semantic_schema_version": 1,
        "semantic_facts": [store.facts[fid].to_dict() for fid in sorted(store.facts)],
    }
    pkl_bytes = pickle.dumps(legacy_graph)
    json_bytes = _json_bytes(restored_doc)

    stage = _make_stage(dest)
    try:
        _write_private(os.path.join(stage, "graph.pkl"), pkl_bytes)
        _write_private(os.path.join(stage, "graph.json"), json_bytes)
        _verify_restored_pair(stage, legacy_graph, store, typed_doc)
        marker = {
            "version": 1,
            "sha256": {
                "graph.pkl": _sha256_bytes(pkl_bytes),
                "graph.json": _sha256_bytes(json_bytes),
            },
            "counts": {"memories": len(legacy_graph.memories), "facts": len(store.facts)},
        }
        _write_private(os.path.join(stage, _MARKER_NAME), _json_bytes(marker))
        if _pair_digests(_read_pair_bytes(legacy_path, "legacy pair")) != legacy_digests:
            raise ValueError("legacy pair changed during the restore")
        for name, digest in bundle_doc["sha256"].items():
            if _sha256_bytes(_read_bundle_file(bundle, name)) != digest:
                raise ValueError("bundle changed during the restore")
        if _sha256_bytes(_read_bundle_file(bundle, "bundle.json")) != marker_doc["sha256"]["bundle.json"]:
            raise ValueError("bundle manifest changed during the restore")
        os.chmod(stage, 0o700)
        _publish_stage(stage, dest)
    except BaseException:
        _cleanup_stage(stage)
        raise


# --------------------------------------------------------------------------- #
# Quarantine
# --------------------------------------------------------------------------- #

def _quarantine_bundle(bundle: str, quarantine_dir: str, reason: str) -> None:
    """Preserve a byte-identical, private copy of a rejected bundle.

    The copy is assembled in a private stage (bundle files first; the
    sanitized reason - field names and fact indices only, never raw memory
    content - written last) and published through the shared atomic no-replace
    boundary, so a quarantine target claimed by a concurrent process is
    refused instead of deleted.
    """
    if os.path.islink(quarantine_dir):
        raise ValueError("quarantine directory must not be a symlink")
    os.makedirs(quarantine_dir, mode=0o700, exist_ok=True)
    os.chmod(quarantine_dir, 0o700)
    name = ("bundle-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-" + secrets.token_hex(6))
    target = os.path.join(quarantine_dir, name)
    stage = _make_stage(target)
    try:
        for entry in sorted(os.listdir(bundle)):
            path = os.path.join(bundle, entry)
            st = os.lstat(path)
            if not stat.S_ISREG(st.st_mode):
                continue
            with open(path, "rb") as fh:
                data = fh.read()
            _write_private(os.path.join(stage, entry), data)
        _write_private(os.path.join(stage, "reason.txt"),
                       (reason + "\n").encode("utf-8"))
        os.chmod(stage, 0o700)
        _publish_stage(stage, target)
    except BaseException:
        _cleanup_stage(stage)
        raise