#!/usr/bin/env python3
"""Strict-validate a pair exactly as its systemd unit would at startup (model override)."""
import sys, os
REL = "/home/dt/.qmg-dk-release/dk-l3-0a09798a23c3"
sys.path.insert(0, REL)
from quantum_memory_graph import rollback as RB

d, model = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
label = os.path.basename(d.rstrip("/"))
try:
    g, doc, pk, js, dig = RB._load_pair_checked(d, label, model=model)
    print("%-16s PASS model=%r mem=%d edges=%d json=%s" % (
        label, model, len(g.memories), g.graph.number_of_edges(),
        "typed" if "semantic_facts" in doc else "legacy"))
except Exception as e:
    print("%-16s FAIL model=%r %s: %s" % (label, model, type(e).__name__, e))
