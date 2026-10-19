#!/usr/bin/env python3
"""End-to-end integrity: source pair vs cold-bootstrapped pair, beyond counts."""
import sys, os, hashlib, pickle

REL = "/home/dt/.qmg-dk-release/dk-l3-0a09798a23c3"
sys.path.insert(0, REL)
from quantum_memory_graph.graph import MemoryGraph
from quantum_memory_graph import rollback as RB

src_dir, dest_dir = sys.argv[1], sys.argv[2]

def fingerprint(g):
    h = hashlib.sha256()
    for mid in sorted(g.memories):
        m = g.memories[mid]
        h.update(mid.encode()); h.update(b"\x00")
        h.update((m.text or "").encode()); h.update(b"\x00")
        h.update((getattr(m, "source", "") or "").encode()); h.update(b"\x01")
    emb = sum(1 for m in g.memories.values() if getattr(m, "embedding", None) is not None)
    ent = sum(len(getattr(m, "entities", []) or []) for m in g.memories.values())
    return {
        "memories": len(g.memories),
        "edges": g.graph.number_of_edges(),
        "text_sha256": h.hexdigest(),
        "embeddings": emb,
        "entity_total": ent,
        "learned_weights": len(getattr(g, "learned_weights", {}) or {}),
        "attrs": len(vars(g)),
        "extra_attrs": sorted(set(vars(g)) - RB._GRAPH_ATTRS),
    }

with open(os.path.join(src_dir, "graph.pkl"), "rb") as f:
    src = pickle.load(f)
dst = MemoryGraph.load(os.path.join(dest_dir, "graph.pkl"))

a, b = fingerprint(src), fingerprint(dst)
print("source:", a)
print("dest  :", b)
same = all(a[k] == b[k] for k in ("memories", "edges", "text_sha256", "embeddings",
                                  "entity_total", "learned_weights"))
print("DATA IDENTICAL:", same, "| dest clean:", b["extra_attrs"] == [])
