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
import pickle
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

# English stopwords for BM25 tokenization
_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'about', 'up', 'down', 'this',
    'that', 'these', 'those', 'it', 'its', 'and', 'but', 'or', 'if',
    'because', 'while', 'which', 'who', 'whom', 'what',
}

def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25: lowercase, split, remove stopwords and short tokens."""
    return [w for w in text.lower().split() if w not in _STOPWORDS and len(w) > 1]

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _HAS_SPACY = True
except (ImportError, OSError):
    _nlp = None
    _HAS_SPACY = False


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

    # Supported models (user can pass any sentence-transformers model name)
    DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    HIGH_ACCURACY_MODEL = 'thenlper/gte-large'  # 96.6% R@5 — tied #1 on LongMemEval

    def __init__(self, similarity_threshold: float = 0.3, model: str = None):
        self.graph = nx.Graph()
        self.memories: Dict[str, Memory] = {}
        self.similarity_threshold = similarity_threshold
        self._model_name = model or self.DEFAULT_MODEL
        self._embedder = None
        self._is_bge = 'bge' in self._model_name.lower()
        self._bm25 = None
        self._bm25_corpus: List[str] = []
        self._entity_index: Dict[str, set] = {}

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._model_name)
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
        
        # Update entity index
        for entity in memory.entities:
            if entity not in self._entity_index:
                self._entity_index[entity] = set()
            self._entity_index[entity].add(mem_id)
        
        # Update BM25 corpus
        self._bm25_corpus.append(text)
        self._bm25 = None  # Invalidate cached BM25
        
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
            
            # Update entity index
            for entity in memory.entities:
                if entity not in self._entity_index:
                    self._entity_index[entity] = set()
                self._entity_index[entity].add(mem_id)
            
            # Update BM25 corpus
            self._bm25_corpus.append(text)
            
            new_memories.append(memory)

        # Invalidate BM25 after batch
        self._bm25 = None

        # Connect all new memories
        for memory in new_memories:
            self._connect_memory(memory)

        return new_memories

    @property
    def _effective_max_connections(self):
        """Dynamic edge cap: 2% of graph size, min 5, max 20."""
        n = len(self.memories)
        return min(20, max(5, int(n * 0.02)))

    def _get_or_build_bm25(self):
        """Lazily build BM25 index from corpus."""
        if not _HAS_BM25:
            return None
        if self._bm25 is not None:
            return self._bm25
        if not self._bm25_corpus:
            return None
        tokenized = [_tokenize(text) for text in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenized)
        return self._bm25

    def _connect_memory(self, memory: Memory):
        """Create edges between this memory and candidate memories."""
        # Candidate set: entity-matched + last 100 recent memories
        candidate_ids: Set[str] = set()

        # Entity-matched candidates
        for entity in memory.entities:
            if entity in self._entity_index:
                candidate_ids.update(self._entity_index[entity])

        # Add last 100 recent memories (by insertion order, as a fallback)
        all_ids = list(self.memories.keys())
        recent_ids = all_ids[-100:]
        candidate_ids.update(recent_ids)

        # Remove self
        candidate_ids.discard(memory.id)

        # Compute weights for all candidates
        scored: List[Tuple[str, float, List[str]]] = []
        for other_id in candidate_ids:
            other = self.memories.get(other_id)
            if other is None:
                continue
            weight, edge_types = self._compute_relationship(memory, other)
            if weight > self.similarity_threshold:
                scored.append((other_id, weight, edge_types))

        # Sort by weight descending, keep top N (dynamic pruning)
        scored.sort(key=lambda x: x[1], reverse=True)
        max_conn = self._effective_max_connections
        for other_id, weight, edge_types in scored[:max_conn]:
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
        """Extract entities using spaCy NER (with fallback to heuristic)."""
        if _HAS_SPACY:
            return self._extract_entities_spacy(text)
        return self._extract_entities_heuristic(text)

    def _extract_entities_spacy(self, text: str) -> List[str]:
        """NER extraction via spaCy — finds people, orgs, products, tech."""
        doc = _nlp(text[:1000])  # Cap for speed
        entities = set()

        # Named entities from spaCy
        for ent in doc.ents:
            if ent.label_ in ('PERSON', 'ORG', 'PRODUCT', 'GPE', 'FAC',
                              'EVENT', 'WORK_OF_ART', 'LAW', 'NORP'):
                entities.add(ent.text.strip())

        # Also extract noun chunks that look like technical terms
        for chunk in doc.noun_chunks:
            # Keep noun chunks with proper nouns or technical terms
            if any(tok.pos_ == 'PROPN' for tok in chunk):
                entities.add(chunk.text.strip())
            elif any(tok.text[0].isupper() and len(tok.text) > 1
                     for tok in chunk if tok.pos_ not in ('DET', 'ADP')):
                clean = ' '.join(tok.text for tok in chunk
                                if tok.pos_ not in ('DET', 'ADP', 'PUNCT'))
                if clean and len(clean) > 1:
                    entities.add(clean.strip())

        # Extract technical terms (CamelCase, acronyms, dash-compounds)
        for tok in doc:
            word = tok.text
            if (len(word) > 1 and word[0].isupper() and
                    any(c.islower() for c in word) and
                    any(c.isupper() for c in word[1:])):
                entities.add(word)  # CamelCase: FastAPI, PostgreSQL
            elif word.isupper() and len(word) >= 2 and word.isalpha():
                entities.add(word)  # Acronyms: API, CI, CD, JWT

        # Filter out common non-entities
        stopwords = {'The', 'This', 'That', 'These', 'Step', 'Phase',
                     'Part', 'Section', 'Note', 'FYI', 'OK', 'IT'}
        return [e for e in entities if e not in stopwords and len(e) > 1]

    def _extract_entities_heuristic(self, text: str) -> List[str]:
        """Fallback: extract capitalized phrases as entities."""
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
                         top_seeds: int = 5) -> Dict[str, float]:
        """
        Find memories related to a query by hybrid BM25+embedding + graph traversal.
        
        1. Score all memories with embedding cosine + BM25 keyword
        2. Normalize each to [0,1], then fuse: 70% embedding + 30% BM25
        3. Select top seeds by fused score
        4. Expand through graph edges (multi-hop)
        5. Return all reachable memory IDs with their connection scores
        """
        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        
        # --- Embedding scores ---
        emb_scores = {}
        for mid, mem in self.memories.items():
            if mem.embedding is not None:
                emb_scores[mid] = float(np.dot(query_emb, mem.embedding))
        
        # --- BM25 scores ---
        bm25_scores = {}
        bm25 = self._get_or_build_bm25()
        if bm25 is not None:
            tokenized_query = _tokenize(query)
            bm25_raw = bm25.get_scores(tokenized_query)
            for i, mid in enumerate(self.memories.keys()):
                if i < len(bm25_raw):
                    bm25_scores[mid] = float(bm25_raw[i])
        
        # --- Fuse: normalize → 70% emb + 30% BM25 ---
        fused = {}
        
        # Normalize embedding scores
        emb_vals = list(emb_scores.values())
        if emb_vals:
            emb_min, emb_max = min(emb_vals), max(emb_vals)
            emb_range = emb_max - emb_min if emb_max != emb_min else 1.0
        else:
            emb_range = 1.0; emb_min = 0.0
        
        # Normalize BM25 scores
        bm25_vals = list(bm25_scores.values())
        if bm25_vals:
            bm25_min, bm25_max = min(bm25_vals), max(bm25_vals)
            bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
        else:
            bm25_range = 1.0; bm25_min = 0.0
        
        for mid in self.memories:
            e_norm = (emb_scores.get(mid, 0.0) - emb_min) / emb_range if emb_range > 0 else 0.0
            if bm25 is None or not bm25_scores:
                fused[mid] = e_norm
            else:
                b_norm = (bm25_scores.get(mid, 0.0) - bm25_min) / bm25_range if bm25_range > 0 else 0.0
                fused[mid] = 0.7 * e_norm + 0.3 * b_norm
        
        # Top seeds by fused score
        seeds = sorted(fused, key=lambda m: fused[m], reverse=True)[:top_seeds]
        
        # Expand through graph
        reachable: Dict[str, float] = {}
        for seed in seeds:
            seed_score = fused[seed]
            reachable[seed] = max(reachable.get(seed, 0), seed_score)
            
            # BFS expansion
            visited = {seed}
            frontier: List[Tuple[str, int]] = [(seed, 0)]
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

    def save(self, path: str):
        """Persist graph to disk as pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str, model: str = None) -> 'MemoryGraph':
        """Load persisted graph from pickle."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        
        # Handle old format: dict with G + memories
        if isinstance(obj, dict) and 'G' in obj and 'memories' in obj:
            from .graph import MemoryGraph as MG
            g = MG(similarity_threshold=0.3, model=model)
            g.graph = obj['G']
            g.memories = obj['memories']
            g._rebuild_indices()
            return g
        
        # New format: serialized MemoryGraph
        if model:
            obj._model_name = model
            obj._embedder = None  # Force re-init on next use
        
        # Rebuild indices if missing (backward compat)
        if not hasattr(obj, '_entity_index') or not obj._entity_index:
            obj._rebuild_indices()
        if not hasattr(obj, '_bm25_corpus'):
            obj._bm25_corpus = []
        if not hasattr(obj, '_bm25'):
            obj._bm25 = None
        
        return obj

    def _rebuild_indices(self):
        """Rebuild entity index and BM25 corpus from existing memories."""
        self._entity_index = {}
        self._bm25_corpus = []
        for mid, mem in self.memories.items():
            for entity in mem.entities:
                if entity not in self._entity_index:
                    self._entity_index[entity] = set()
                self._entity_index[entity].add(mid)
            self._bm25_corpus.append(mem.text)
        self._bm25 = None
