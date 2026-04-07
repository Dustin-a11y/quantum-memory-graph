"""
Quantum Memory Graph — Relationship-aware memory for AI agents.

Uses knowledge graphs to map relationships between memories,
then QAOA to find the optimal subgraph for any query.

Copyright 2026 Coinkong (Chef's Attraction). MIT License.
"""

__version__ = "0.1.0"

from .graph import MemoryGraph
from .subgraph_optimizer import optimize_subgraph
from .pipeline import recall, store

__all__ = ["MemoryGraph", "optimize_subgraph", "recall", "store"]
