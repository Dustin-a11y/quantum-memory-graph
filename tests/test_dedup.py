"""
Tests for memory deduplication engine.
DK 🦍 — RED → GREEN → SHIP
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from quantum_memory_graph.dedup import MemoryDeduplicator, DuplicateGroup, deduplicate
from quantum_memory_graph.graph import MemoryGraph, Memory


class TestMemoryDeduplicator:
    def _make_graph_with_dupes(self):
        """Create a graph with known duplicates."""
        graph = MemoryGraph.__new__(MemoryGraph)
        graph.graph = __import__('networkx').Graph()
        graph.memories = {}
        graph.similarity_threshold = 0.3
        
        # Create "duplicate" memories with identical embeddings
        emb1 = np.random.randn(384)
        emb1 = emb1 / np.linalg.norm(emb1)
        
        m1 = Memory(id="m1", text="User prefers dark mode", embedding=emb1.copy(),
                     entities=["user", "dark mode"], timestamp=datetime.now())
        m2 = Memory(id="m2", text="User prefers dark mode setting", embedding=emb1.copy(),
                     entities=["user"], timestamp=datetime.now() - timedelta(hours=1))
        
        # Unique memory
        emb3 = np.random.randn(384)
        emb3 = emb3 / np.linalg.norm(emb3)
        m3 = Memory(id="m3", text="Server runs on port 8080", embedding=emb3,
                     entities=["server"], timestamp=datetime.now())
        
        graph.memories = {"m1": m1, "m2": m2, "m3": m3}
        graph.graph.add_nodes_from(["m1", "m2", "m3"])
        
        return graph
    
    def test_find_duplicates(self):
        graph = self._make_graph_with_dupes()
        deduper = MemoryDeduplicator(threshold=0.99)
        groups = deduper.find_duplicates(graph)
        
        assert len(groups) == 1, f"Should find 1 duplicate group, found {len(groups)}"
        assert "m3" not in groups[0].duplicate_ids, "Unique memory should not be duplicate"
    
    def test_canonical_picks_richest(self):
        graph = self._make_graph_with_dupes()
        deduper = MemoryDeduplicator(threshold=0.99)
        groups = deduper.find_duplicates(graph)
        
        # m1 has more entities than m2, should be canonical
        assert groups[0].canonical_id == "m1", "Should keep memory with more entities"
    
    def test_merge_removes_duplicates(self):
        graph = self._make_graph_with_dupes()
        deduper = MemoryDeduplicator(threshold=0.99)
        
        assert len(graph.memories) == 3
        stats = deduper.merge_duplicates(graph)
        assert stats["duplicates_removed"] >= 1
        assert len(graph.memories) < 3
    
    def test_dry_run_doesnt_modify(self):
        graph = self._make_graph_with_dupes()
        deduper = MemoryDeduplicator(threshold=0.99)
        
        original_count = len(graph.memories)
        stats = deduper.merge_duplicates(graph, dry_run=True)
        assert len(graph.memories) == original_count, "Dry run should not modify graph"
        assert stats["duplicates_removed"] >= 1, "Should still report what would be removed"
    
    def test_no_duplicates_below_threshold(self):
        graph = self._make_graph_with_dupes()
        deduper = MemoryDeduplicator(threshold=1.01)  # Impossible threshold
        groups = deduper.find_duplicates(graph)
        assert len(groups) == 0
    
    def test_entity_merge(self):
        graph = self._make_graph_with_dupes()
        deduper = MemoryDeduplicator(threshold=0.99)
        stats = deduper.merge_duplicates(graph)
        
        # After merge, canonical should have entities from both
        canonical = graph.memories.get("m1")
        assert canonical is not None
        assert "dark mode" in canonical.entities or "user" in canonical.entities


class TestDedupEmpty:
    def test_empty_graph(self):
        graph = MemoryGraph.__new__(MemoryGraph)
        graph.graph = __import__('networkx').Graph()
        graph.memories = {}
        graph.similarity_threshold = 0.3
        
        deduper = MemoryDeduplicator()
        groups = deduper.find_duplicates(graph)
        assert len(groups) == 0
    
    def test_single_memory(self):
        graph = MemoryGraph.__new__(MemoryGraph)
        graph.graph = __import__('networkx').Graph()
        graph.memories = {}
        graph.similarity_threshold = 0.3
        
        emb = np.random.randn(384)
        emb = emb / np.linalg.norm(emb)
        m = Memory(id="m1", text="Solo memory", embedding=emb)
        graph.memories = {"m1": m}
        
        deduper = MemoryDeduplicator()
        groups = deduper.find_duplicates(graph)
        assert len(groups) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
