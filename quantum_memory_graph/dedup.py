"""
Memory Deduplication Engine.

Finds and merges duplicate/near-duplicate memories using cosine similarity.
Keeps the richest version (most entities, most recent timestamp).

DK 🦍 — v1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

from .graph import MemoryGraph, Memory


@dataclass
class DuplicateGroup:
    """A group of duplicate memories."""
    canonical_id: str  # The one to keep
    duplicate_ids: List[str]  # The ones to merge/remove
    similarity: float  # Average pairwise similarity


class MemoryDeduplicator:
    """
    Find and merge duplicate memories in a MemoryGraph.
    
    Two memories are duplicates if cosine similarity > threshold.
    When merging, we keep the one with:
      1. More entities (richer knowledge)
      2. More recent timestamp (fresher)
      3. Longer text (more detail)
    """
    
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
    
    def find_duplicates(self, graph: MemoryGraph) -> List[DuplicateGroup]:
        """
        Scan all memory pairs and find duplicate groups.
        
        Returns list of DuplicateGroup, each with a canonical (keep)
        and list of duplicates (remove/merge).
        """
        memories = list(graph.memories.values())
        n = len(memories)
        
        if n < 2:
            return []
        
        # Build similarity matrix
        embeddings = np.array([m.embedding for m in memories if m.embedding is not None])
        if len(embeddings) < 2:
            return []
        
        # Cosine similarity matrix (embeddings are normalized)
        sim_matrix = embeddings @ embeddings.T
        
        # Find pairs above threshold
        visited = set()
        groups = []
        
        for i in range(n):
            if i in visited:
                continue
            
            cluster = [i]
            for j in range(i + 1, n):
                if j in visited:
                    continue
                if sim_matrix[i][j] >= self.threshold:
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) > 1:
                visited.add(i)
                # Pick canonical: most entities > most recent > longest text
                canonical_idx = self._pick_canonical(memories, cluster)
                duplicate_idxs = [idx for idx in cluster if idx != canonical_idx]
                
                avg_sim = np.mean([
                    sim_matrix[canonical_idx][d] for d in duplicate_idxs
                ])
                
                groups.append(DuplicateGroup(
                    canonical_id=memories[canonical_idx].id,
                    duplicate_ids=[memories[d].id for d in duplicate_idxs],
                    similarity=float(avg_sim),
                ))
        
        return groups
    
    def _pick_canonical(self, memories: List[Memory], indices: List[int]) -> int:
        """Pick the best memory to keep from a duplicate group."""
        scored = []
        for idx in indices:
            m = memories[idx]
            score = (
                len(m.entities) * 10 +  # More entities = richer
                (1 if m.timestamp else 0) * 5 +  # Has timestamp
                len(m.text) * 0.01  # Longer = more detail
            )
            # Prefer newer
            if m.timestamp:
                score += m.timestamp.timestamp() / 1e10
            scored.append((idx, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    
    def merge_duplicates(self, graph: MemoryGraph, 
                         groups: List[DuplicateGroup] = None,
                         dry_run: bool = False) -> Dict:
        """
        Merge duplicate groups in the graph.
        
        For each group:
          1. Keep the canonical memory
          2. Merge entities from duplicates into canonical
          3. Remove duplicate nodes and edges
        
        Returns stats about what was merged.
        """
        if groups is None:
            groups = self.find_duplicates(graph)
        
        stats = {
            "groups_found": len(groups),
            "memories_before": len(graph.memories),
            "duplicates_removed": 0,
            "entities_merged": 0,
            "details": [],
        }
        
        if dry_run:
            for g in groups:
                canonical = graph.memories.get(g.canonical_id)
                dupes = [graph.memories.get(d) for d in g.duplicate_ids if d in graph.memories]
                stats["details"].append({
                    "keep": canonical.text[:80] if canonical else "?",
                    "remove": [d.text[:80] for d in dupes if d],
                    "similarity": round(g.similarity, 3),
                })
            stats["duplicates_removed"] = sum(len(g.duplicate_ids) for g in groups)
            return stats
        
        for group in groups:
            canonical = graph.memories.get(group.canonical_id)
            if not canonical:
                continue
            
            # Merge entities from duplicates
            all_entities = set(canonical.entities)
            for dup_id in group.duplicate_ids:
                dup = graph.memories.get(dup_id)
                if dup:
                    all_entities.update(dup.entities)
                    # Remove duplicate from graph
                    if dup_id in graph.graph:
                        graph.graph.remove_node(dup_id)
                    del graph.memories[dup_id]
                    stats["duplicates_removed"] += 1
            
            # Update canonical with merged entities
            new_entities = list(all_entities)
            merged_count = len(new_entities) - len(canonical.entities)
            canonical.entities = new_entities
            stats["entities_merged"] += max(0, merged_count)
            
            stats["details"].append({
                "keep": canonical.text[:80],
                "removed": len(group.duplicate_ids),
                "similarity": round(group.similarity, 3),
                "entities_gained": max(0, merged_count),
            })
        
        stats["memories_after"] = len(graph.memories)
        return stats


def deduplicate(graph: MemoryGraph = None, threshold: float = 0.95,
                dry_run: bool = False) -> Dict:
    """
    Convenience function — find and merge duplicates.
    
    Usage:
        from quantum_memory_graph import deduplicate
        stats = deduplicate(threshold=0.95)  # Uses default graph
        stats = deduplicate(threshold=0.90, dry_run=True)  # Preview only
    """
    from .pipeline import get_graph
    g = graph or get_graph()
    deduper = MemoryDeduplicator(threshold=threshold)
    return deduper.merge_duplicates(g, dry_run=dry_run)
