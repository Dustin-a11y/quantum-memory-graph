"""
Knowledge Graph for Memory Relationships.

Builds a graph where:
- Nodes = individual memories
- Edges = relationships (semantic similarity, temporal proximity,
  entity co-occurrence, causal links)
- Edge weights = relationship strength

This is what makes us different from every other memory system.
They treat memories as independent documents. We treat them as a network.

Copyright 2026 Coinkong (Chef's Attraction). MIT License.
"""

import hashlib
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Memory:
    """A single memory node in the graph."""
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    entities: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    source: str = ""
    metadata: Dict = field(default_factory=dict)

    @staticmethod
    def make_id(text: str) -> str:
        return hashlib.md5(text[:200].encode()).hexdigest()[:12]


class MemoryGraph:
    """
    Knowledge graph of memory relationships.
    
    Memories are nodes. Relationships are weighted edges.
    Edge types:
      - semantic: embedding cosine similarity
      - temporal: time proximity (memories close in time are related)
      - entity: shared entities (people, places, projects)
      - causal: A references or builds on B
    """

    def __init__(self, similarity_threshold: float = 0.3):
        self.graph = nx.Graph()
        self.memories: Dict[str, Memory] = {}
        self.similarity_threshold = similarity_threshold
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedder

    def add_memory(self, text: str, entities: List[str] = None,
                   timestamp: datetime = None, source: str = "",
                   metadata: Dict = None) -> Memory:
        """Add a memory and connect it to existing memories."""
        mem_id = Memory.make_id(text)
        
        if mem_id in self.memories:
            return self.memories[mem_id]

        embedding = self.embedder.encode([text], normalize_embeddings=True)[0]
        
        entities = entities or self._extract_entities(text)
        
        memory = Memory(
            id=mem_id,
            text=text,
            embedding=embedding,
            entities=entities,
            timestamp=timestamp or datetime.now(),
            source=source,
            metadata=metadata or {},
        )
        
        self.memories[mem_id] = memory
        self.graph.add_node(mem_id, memory=memory)
        
        # Connect to existing memories
        self._connect_memory(memory)
        
        return memory

    def add_memories_batch(self, texts: List[str],
                           entities_list: List[List[str]] = None,
                           timestamps: List[datetime] = None,
                           sources: List[str] = None) -> List[Memory]:
        """Batch add memories (faster embedding)."""
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        
        new_memories = []
        for i, text in enumerate(texts):
            mem_id = Memory.make_id(text)
            if mem_id in self.memories:
                new_memories.append(self.memories[mem_id])
                continue

            entities = (entities_list[i] if entities_list else
                       self._extract_entities(text))
            timestamp = (timestamps[i] if timestamps else datetime.now())
            source = (sources[i] if sources else "")

            memory = Memory(
                id=mem_id, text=text, embedding=embeddings[i],
                entities=entities, timestamp=timestamp, source=source,
            )
            self.memories[mem_id] = memory
            self.graph.add_node(mem_id, memory=memory)
            new_memories.append(memory)

        # Connect all new memories
        for memory in new_memories:
            self._connect_memory(memory)

        return new_memories

    def _connect_memory(self, memory: Memory):
        """Create edges between this memory and all related memories."""
        for other_id, other in self.memories.items():
            if other_id == memory.id:
                continue

            weight, edge_types = self._compute_relationship(memory, other)
            
            if weight > self.similarity_threshold:
                self.graph.add_edge(
                    memory.id, other_id,
                    weight=weight,
                    types=edge_types,
                )

    def _compute_relationship(self, a: Memory, b: Memory) -> Tuple[float, List[str]]:
        """
        Compute multi-dimensional relationship strength between two memories.
        Returns (weight, list_of_relationship_types).
        """
        weight = 0.0
        types = []

        # 1. Semantic similarity (cosine)
        if a.embedding is not None and b.embedding is not None:
            sim = float(np.dot(a.embedding, b.embedding))
            if sim > 0.2:
                weight += sim * 0.4  # 40% weight
                types.append("semantic")

        # 2. Entity co-occurrence
        if a.entities and b.entities:
            shared = set(a.entities) & set(b.entities)
            if shared:
                entity_score = len(shared) / max(len(a.entities), len(b.entities))
                weight += entity_score * 0.35  # 35% weight
                types.append("entity")

        # 3. Temporal proximity
        if a.timestamp and b.timestamp:
            delta = abs((a.timestamp - b.timestamp).total_seconds())
            # Within 1 hour = strong, within 1 day = moderate, beyond = weak
            if delta < 3600:
                temporal = 0.9
            elif delta < 86400:
                temporal = 0.5
            elif delta < 604800:
                temporal = 0.2
            else:
                temporal = 0.05
            weight += temporal * 0.15  # 15% weight
            types.append("temporal")

        # 4. Source proximity (same conversation/document)
        if a.source and b.source and a.source == b.source:
            weight += 0.1  # 10% weight
            types.append("source")

        return weight, types

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction (proper nouns, capitalized phrases)."""
        entities = []
        words = text.split()
        i = 0
        while i < len(words):
            word = words[i].strip('.,!?:;()[]{}"\'-')
            if (word and word[0].isupper() and len(word) > 1
                    and word not in ('I', 'The', 'A', 'An', 'In', 'On', 'At',
                                    'For', 'To', 'Of', 'And', 'But', 'Or',
                                    'Is', 'Was', 'Are', 'Were', 'It', 'This',
                                    'That', 'With', 'From', 'By', 'As', 'If',
                                    'Not', 'No', 'Yes', 'So', 'Do', 'Did',
                                    'Has', 'Had', 'Have', 'Be', 'Been',
                                    'Will', 'Would', 'Could', 'Should',
                                    'May', 'Can', 'We', 'He', 'She', 'They',
                                    'My', 'His', 'Her', 'Our', 'Your',
                                    'What', 'When', 'Where', 'How', 'Why',
                                    'Who', 'Which')):
                # Collect multi-word entity
                entity_parts = [word]
                j = i + 1
                while j < len(words):
                    next_word = words[j].strip('.,!?:;()[]{}"\'-')
                    if next_word and next_word[0].isupper() and len(next_word) > 1:
                        entity_parts.append(next_word)
                        j += 1
                    else:
                        break
                entities.append(" ".join(entity_parts))
                i = j
            else:
                i += 1
        return list(set(entities))

    def get_neighborhood(self, query: str, hops: int = 2,
                         top_seeds: int = 5) -> List[str]:
        """
        Find memories related to a query by graph traversal.
        
        1. Find top seed memories by embedding similarity
        2. Expand through graph edges (multi-hop)
        3. Return all reachable memory IDs with their connection scores
        """
        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        
        # Score all memories against query
        scores = {}
        for mid, mem in self.memories.items():
            if mem.embedding is not None:
                scores[mid] = float(np.dot(query_emb, mem.embedding))
        
        # Top seeds
        seeds = sorted(scores, key=scores.get, reverse=True)[:top_seeds]
        
        # Expand through graph
        reachable = {}
        for seed in seeds:
            seed_score = scores[seed]
            reachable[seed] = max(reachable.get(seed, 0), seed_score)
            
            # BFS expansion
            visited = {seed}
            frontier = [(seed, 0)]
            while frontier:
                node, depth = frontier.pop(0)
                if depth >= hops:
                    continue
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        edge_w = self.graph[node][neighbor].get('weight', 0)
                        # Score decays with hops but boosted by edge weight
                        hop_score = seed_score * edge_w * (0.7 ** (depth + 1))
                        reachable[neighbor] = max(
                            reachable.get(neighbor, 0), hop_score
                        )
                        frontier.append((neighbor, depth + 1))
        
        return reachable

    def get_subgraph_data(self, node_ids: List[str]) -> Dict:
        """
        Extract adjacency and weight data for QAOA optimization.
        Returns node list, adjacency matrix, and relevance scores.
        """
        nodes = [nid for nid in node_ids if nid in self.graph]
        n = len(nodes)
        adj = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.graph.has_edge(nodes[i], nodes[j]):
                    w = self.graph[nodes[i]][nodes[j]].get('weight', 0)
                    adj[i][j] = w
                    adj[j][i] = w
        
        return {
            "nodes": nodes,
            "adjacency": adj,
            "memories": [self.memories[nid] for nid in nodes],
        }

    def stats(self) -> Dict:
        """Graph statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
            "components": nx.number_connected_components(self.graph),
            "avg_degree": (2 * self.graph.number_of_edges() / 
                          max(self.graph.number_of_nodes(), 1)),
        }
