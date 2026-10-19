"""
Self-learning graph — adaptive edge weights and memory value scoring.
Learns from recall patterns: edges between frequently co-recalled memories get stronger.
"""
import itertools
from collections import defaultdict
from typing import List, Set, Dict, Tuple
import json
import os

class RecallTracker:
    """Tracks recall patterns to learn edge weights."""
    
    def __init__(self, graph_dir: str = ""):
        self.history = []  # List of (query, [node_ids])
        self.graph_dir = graph_dir
        self.co_recall_count = defaultdict(float)  # (node_a, node_b) -> count
    
    def log_recall(self, query: str, node_ids: List[str], scores: Dict[str, float] | None = None):
        """Record a recall event — which nodes were retrieved for a query."""
        entry = {
            "query": query[:200],
            "node_ids": node_ids,
            "count": len(node_ids),
        }
        self.history.append(entry)
        # Update co-recall counts
        for a, b in itertools.combinations(sorted(node_ids), 2):
            self.co_recall_count[(a, b)] += 1.0
    
    def learn_weights(self, min_co_recalls: int = 2) -> Dict[Tuple[str, str], float]:
        """
        Compute learned edge weights from recall history.
        Returns {(node_a, node_b): weight} for edges with >= min_co_recalls.
        Weights are normalized 0.0-1.0.
        """
        if not self.co_recall_count:
            return {}
        max_count = max(self.co_recall_count.values())
        weights = {}
        for (a, b), count in self.co_recall_count.items():
            if count >= min_co_recalls:
                # Log-scaled, normalized weight
                weights[(a, b)] = min(1.0, count / max(1, max_count))
        return weights
    
    def get_memory_values(self) -> Dict[str, float]:
        """Score each memory by how often it's recalled. 0.0 = never, 1.0 = most recalled."""
        recall_count = defaultdict(int)
        for entry in self.history:
            for nid in entry["node_ids"]:
                recall_count[nid] += 1
        if not recall_count:
            return {}
        max_c = max(recall_count.values())
        return {nid: min(1.0, c / max(1, max_c)) for nid, c in recall_count.items()}
    
    def save(self, path: str):
        """Persist recall history to JSON."""
        data = {
            "co_recall_count": {f"{a}|||{b}": c for (a,b), c in self.co_recall_count.items()},
            "history_len": len(self.history),
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load recall history from JSON."""
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for key, count in data.get("co_recall_count", {}).items():
                a, b = key.split("|||")
                self.co_recall_count[(a, b)] = count
