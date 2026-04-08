"""
Quantum Memory Graph — Relationship-aware memory for AI agents.

Uses knowledge graphs to map relationships between memories,
then QAOA to find the optimal subgraph for any query.

Copyright 2026 Coinkong (Chef's Attraction). MIT License.
"""

__version__ = "0.4.0"

from .graph import MemoryGraph
from .subgraph_optimizer import optimize_subgraph
from .pipeline import recall, store, get_stm, set_stm
from .recency import ShortTermMemory, RecencyBooster, WorkingMemory, ConversationContext

__all__ = [
    "MemoryGraph", "optimize_subgraph", "recall", "store",
    "ShortTermMemory", "RecencyBooster", "WorkingMemory", "ConversationContext",
    "get_stm", "set_stm",
]
