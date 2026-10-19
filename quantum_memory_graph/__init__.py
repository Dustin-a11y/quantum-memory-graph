"""
Quantum Memory Graph — Relationship-aware memory for AI agents.

Uses knowledge graphs to map relationships between memories,
then QAOA to find the optimal subgraph for any query.

Copyright 2026 Coinkong (Chef's Attraction). MIT License.
"""

__version__ = "1.3.0"

from .graph import MemoryGraph
from .semantic import SemanticFact, AmbiguousEntityError
from .subgraph_optimizer import optimize_subgraph
from .pipeline import recall, recall_facts, store, get_stm, set_stm
from .recency import ShortTermMemory, RecencyBooster, WorkingMemory, ConversationContext

__all__ = [
    "MemoryGraph", "SemanticFact", "AmbiguousEntityError",
    "optimize_subgraph", "recall", "recall_facts", "store",
    "ShortTermMemory", "RecencyBooster", "WorkingMemory", "ConversationContext",
    "get_stm", "set_stm",
]
