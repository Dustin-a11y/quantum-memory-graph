"""
Quantum Memory Graph — Relationship-aware memory for AI agents.

Uses knowledge graphs to map relationships between memories,
then QAOA to find the optimal subgraph for any query.

v1.0.0 — Full system:
  - Knowledge graph with multi-hop traversal
  - QAOA subgraph optimization  
  - Short-term memory (recency, working memory, conversation context)
  - Memory deduplication
  - Semantic memory tiers (hot/warm/cold)
  - Cross-agent memory sharing
  - Obsidian vault export

Copyright 2026 Coinkong (Chef's Attraction). MIT License.
"""

__version__ = "1.0.0"

from .graph import MemoryGraph
from .subgraph_optimizer import optimize_subgraph
from .pipeline import recall, store, get_stm, set_stm
from .recency import ShortTermMemory, RecencyBooster, WorkingMemory, ConversationContext
from .dedup import MemoryDeduplicator, deduplicate
from .tiers import MemoryTierManager, HotTier, WarmTier, TieredMemory
from .sharing import SharedMemoryPool, SharedMemory
from .obsidian import export_vault, export_from_mem0

__all__ = [
    # Core
    "MemoryGraph", "optimize_subgraph", "recall", "store",
    # Short-term memory
    "ShortTermMemory", "RecencyBooster", "WorkingMemory", "ConversationContext",
    "get_stm", "set_stm",
    # Deduplication
    "MemoryDeduplicator", "deduplicate",
    # Tiered memory
    "MemoryTierManager", "HotTier", "WarmTier", "TieredMemory",
    # Cross-agent sharing
    "SharedMemoryPool", "SharedMemory",
    # Obsidian
    "export_vault", "export_from_mem0",
]
