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
from datetime import datetime, timezone
from .learning import RecallTracker
from .semantic import SemanticStore

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
    temporal_entities: List[Dict] = field(default_factory=list)

    @staticmethod
    def make_id(text: str) -> str:
        return hashlib.md5(text[:200].encode()).hexdigest()[:12]


# Attributes written by third-generation pickles that no current code reads.
# They are vestigial (sources/_source_index were always empty, max_connections
# was always the default 50), but rollback's strict pair validator rejects any
# surviving unknown attribute, so a pair saved by this package would be
# rejected by its own validator. load() strips them under guard so old pairs
# converge on the next save; anything carrying data fails closed instead.
_LEGACY_GRAPH_ATTRS = ("sources", "_source_index", "max_connections")


def _strip_legacy_graph_attrs(obj):
    """Remove third-generation attributes that no current code reads.

    Refuses to drop anything that carries data. Only the three names in
    ``_LEGACY_GRAPH_ATTRS`` are ever touched; every other attribute is left
    for the caller's validator to judge.
    """
    for attr in _LEGACY_GRAPH_ATTRS:
        if not hasattr(obj, attr):
            continue
        value = getattr(obj, attr)
        if attr == "max_connections":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    'legacy max_connections is not a plain int; refusing to drop')
        else:
            try:
                empty = len(value) == 0
            except TypeError:
                raise ValueError(
                    f'legacy {attr} is not a sized container; refusing to drop')
            if not empty:
                raise ValueError(f'legacy {attr} is non-empty; refusing data loss')
        delattr(obj, attr)
    return obj


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
        # Opt-in typed assertions remain separate from legacy text nodes.
        self.semantic_store = SemanticStore()
        
        # Self-learning: recall pattern tracking and learned edge weights
        self.recall_tracker = RecallTracker("")
        self.learned_weights: Dict[Tuple[str, str], float] = {}

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._model_name)
        return self._embedder

    @staticmethod
    def _normalize_timestamp(timestamp: Optional[datetime]) -> Optional[datetime]:
        """Return timestamps as UTC-naive for consistent temporal arithmetic."""
        if timestamp is not None and timestamp.tzinfo is not None:
            return timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        return timestamp

    def add_memory(self, text: str, entities: List[str] = None,
                   timestamp: datetime = None, source: str = "",
                   metadata: Dict = None, semantic_facts: List[Dict] = None) -> Memory:
        """Add a memory and connect it to existing memories."""
        mem_id = Memory.make_id(text)
        if semantic_facts is not None and not isinstance(semantic_facts, (list, tuple)):
            raise ValueError("semantic_facts must be a list of explicit assertions")
        # Validate the entire typed write before changing the legacy graph.
        prepared = (SemanticStore.prepare(semantic_facts, source, mem_id)
                    if semantic_facts else [])
        
        if mem_id in self.memories:
            existing = self.memories[mem_id]
            if prepared:
                # Text-only IDs can dedup a typed write onto a memory that
                # belongs to a different source (or, on a hash-prefix
                # collision, different text). Until evidence records are
                # source-qualified, fail closed before any mutation.
                if getattr(existing, "source", "") != source:
                    raise ValueError(
                        "typed write would attach to an existing memory from a "
                        "different source; refusing cross-source provenance"
                    )
                if getattr(existing, "text", None) != text:
                    raise ValueError(
                        "typed write collides with a different memory text; "
                        "refusing ambiguous provenance"
                    )
                self.semantic_store.commit(prepared)
            return existing

        embedding = self.embedder.encode([text], normalize_embeddings=True)[0]
        
        entities = entities or self._extract_entities(text)
        
        temporal = self._extract_temporal_entities(text)
        if timestamp is not None and timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        memory = Memory(
            id=mem_id,
            text=text,
            embedding=embedding,
            entities=entities,
            timestamp=timestamp or datetime.now(),
            source=source,
            metadata=metadata or {},
            temporal_entities=temporal,
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
        if prepared:
            self.semantic_store.commit(prepared)
        
        return memory

    def query_facts(self, subject_id: str, relation_type: str,
                    observed_as_of: datetime = None,
                    valid_at: datetime = None):
        """Exact indexed typed lookup; return all versions with provenance."""
        return self.semantic_store.query(subject_id, relation_type,
                                         observed_as_of=observed_as_of,
                                         valid_at=valid_at)

    def resolve_entity(self, label: str, entity_type: str = None) -> str:
        """Resolve only unique display labels; ambiguous names require IDs."""
        return self.semantic_store.resolve_entity(label, entity_type=entity_type)

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
            if timestamp.tzinfo is not None:
                timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
            source = (sources[i] if sources else "")

            memory = Memory(
                id=mem_id, text=text, embedding=embeddings[i],
                entities=entities, timestamp=timestamp, source=source,
                temporal_entities=self._extract_temporal_entities(text),
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

    def _effective_max_connections(self):
        """Dynamic edge cap: 2% of graph size, min 5, max overridable."""
        import os
        env_max = os.environ.get("QMG_MAX_CONNECTIONS", "")
        if env_max:
            return int(env_max)
        n = len(self.memories)
        return min(20, max(5, int(n * 0.02)))

    def _assert_removal_safe(self, memory_ids) -> None:
        """Leave typed evidence intact until Layer 5 defines retraction."""
        targets = set(memory_ids)
        if not targets or not self.semantic_store.facts:
            return
        for fact in self.semantic_store.facts.values():
            evidence = fact.evidence_memory_ids or (fact.source_memory_id,)
            if targets.intersection(evidence):
                raise ValueError("memory links to a typed semantic fact; explicit retraction required")

    def _remove_memory(self, mem_id: str) -> bool:
        """Remove a memory node and all its references from the graph structures."""
        if mem_id not in self.memories:
            return False
        self._assert_removal_safe((mem_id,))
        memory = self.memories[mem_id]
        # Remove from entity index
        for entity in memory.entities:
            if entity in self._entity_index:
                self._entity_index[entity].discard(mem_id)
                if not self._entity_index[entity]:
                    del self._entity_index[entity]
        # Remove from BM25 corpus (rebuild-safe: drop by text match)
        try:
            self._bm25_corpus.remove(memory.text)
        except ValueError:
            # Text was not in the BM25 corpus; dropping the node is still correct.
            pass
        self._bm25 = None  # Invalidate cached BM25
        # Remove node from graph (drops its edges)
        if self.graph.has_node(mem_id):
            self.graph.remove_node(mem_id)
        # Remove from memories dict
        del self.memories[mem_id]
        return True

    def delete_by_source(self, sources: List[str], dry_run: bool = False) -> Dict:
        """Delete memories whose source matches any in `sources`.
        Exact match or prefix match (e.g. 'benchd-session' matches
        'benchd-session-0'). dry_run reports what would be removed
        without changing the graph. Returns aggregate stats."""
        targets = []
        for mid, mem in list(self.memories.items()):
            src = getattr(mem, "source", "") or ""
            for s in sources:
                if src == s or src.startswith(s):
                    targets.append((mid, src))
                    break
        if not dry_run:
            self._assert_removal_safe(mid for mid, _ in targets)
        removed = 0
        by_source: Dict[str, int] = {}
        for mid, src in targets:
            by_source[src] = by_source.get(src, 0) + 1
            if not dry_run:
                if self._remove_memory(mid):
                    removed += 1
        if dry_run:
            removed = len(targets)
        return {
            "ok": True,
            "matched": len(targets),
            "removed": removed,
            "dry_run": dry_run,
            "by_source": dict(sorted(by_source.items(), key=lambda x: -x[1])[:20]),
            "remaining": len(self.memories),
        }

    def merge_memories(self, memory_ids: List[str]) -> Dict:
        """Merge a cluster of duplicate memories into a single survivor.

        Survivor selection: the memory with the highest graph degree (most
        connected); ties broken by longest text (most informative), then
        earliest timestamp (canonical first occurrence).

        The survivor keeps its node + edges; losers are removed from
        memories, graph, entity index, and BM25 corpus. Loser edges are NOT
        redirected — the survivor already has its own edges, and spurious
        inherited edges would corrupt the graph. Returns a result dict.
        """
        if not memory_ids or len(memory_ids) < 2:
            return {"ok": False, "error": "Need at least 2 memory IDs to merge"}
        present = [m for m in memory_ids if m in self.memories]
        if len(present) < 2:
            return {"ok": False, "error": f"Only {len(present)} of {len(memory_ids)} IDs exist in graph"}

        # Conservative pre-Layer-5 gate: every present cluster member (the
        # would-be survivor included) must be free of typed evidence links, so
        # direct merges match merge_clusters' batch-wide preflight. This runs
        # before survivor selection and before any node is removed.
        self._assert_removal_safe(present)

        def _score(mid: str):
            mem = self.memories[mid]
            degree = self.graph.degree(mid) if self.graph.has_node(mid) else 0
            ts = mem.timestamp or datetime.now()
            return (degree, len(mem.text or ""), -ts.timestamp() if hasattr(ts, 'timestamp') else 0)

        survivor = max(present, key=_score)
        losers = [m for m in present if m != survivor]
        removed = 0
        for mid in losers:
            if self._remove_memory(mid):
                removed += 1
        surv_mem = self.memories.get(survivor)
        return {
            "ok": True,
            "survivor": survivor,
            "survivor_text": (surv_mem.text[:120] if surv_mem else ""),
            "merged": removed,
            "requested": len(memory_ids),
        }

    def merge_clusters(self, clusters: List[List[str]], dry_run: bool = False) -> Dict:
        """Merge many duplicate clusters. Returns aggregate stats.
        dry_run: report what would merge without changing the graph."""
        total_merged = 0
        total_clusters = 0
        survivors = []
        if not dry_run:
            # Conservative batch-wide gate: do not partially merge earlier
            # clusters before discovering a later typed evidence link.
            self._assert_removal_safe(
                mid for cluster in clusters for mid in cluster if mid in self.memories
            )
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            present = [m for m in cluster if m in self.memories]
            if len(present) < 2:
                continue
            if dry_run:
                survivor = present[0]
                total_merged += len(present) - 1
                total_clusters += 1
                survivors.append(survivor)
            else:
                res = self.merge_memories(present)
                if res.get("ok"):
                    total_merged += res["merged"]
                    total_clusters += 1
                    survivors.append(res["survivor"])
        return {
            "ok": True,
            "clusters_merged": total_clusters,
            "memories_merged": total_merged,
            "dry_run": dry_run,
            "survivors": survivors[:50],  # sample
        }

    def prune_existing(self, max_connections: int = None) -> Dict:
        """Prune existing graph edges to max_connections per node.
        
        Keeps only the strongest edges per node. If max_connections not
        specified, uses _effective_max_connections().
        """
        max_conn = max_connections or self._effective_max_connections()
        removed = 0
        kept = 0
        
        for node_id in list(self.graph.nodes()):
            edges = list(self.graph.edges(node_id, data=True))
            if len(edges) <= max_conn:
                kept += len(edges)
                continue
            
            # Sort by weight descending, keep top N
            edges.sort(key=lambda x: x[2].get("weight", 0), reverse=True)
            for _, neighbor, _ in edges[max_conn:]:
                self.graph.remove_edge(node_id, neighbor)
                removed += 1
            kept += max_conn
        
        return {"pruned_edges": removed, "kept_edges": kept, "max_per_node": max_conn}

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
        max_conn = self._effective_max_connections()
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

    def _extract_temporal_entities(self, text: str) -> List[Dict]:
        """Extract dates, ages, durations from text for temporal reasoning."""
        import re
        temporal = []
        
        # --- Dates (absolute) ---
        # "February 20th, 2024", "Feb 20", "2024-02-20", "02/20/2024"
        date_patterns = [
            (r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})', 
             lambda m: f"{m.group(3)}-{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'].index(m.group(1)[:3])+1:02d}-{int(m.group(2)):02d}"),
            (r'(\d{4})-(\d{2})-(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
        ]
        for pattern, fmt in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    iso = fmt(match)
                    temporal.append({"type": "date", "value": iso, "text": match.group(0)})
                except:
                    # A malformed date token is skipped; other temporal entities are still kept.
                    pass

        # --- Ages ("Rachel is 12", "I'm 25 years old") ---
        # Length caps are deliberate: unbounded \w+ / \d+ quantifiers ahead of a
        # required \s+ backtrack quadratically on long single-token input (e.g. a
        # 50k-digit memory text blocked extraction for ~115s). Capping each
        # quantifier bounds the engine to a constant number of splits per
        # position, making the scan linear on adversarial input.
        for m in re.finditer(r'(\w{1,40}(?:\s+\w{1,40})?)\s+(?:is|was|will be)\s+(\d{1,3})\s*(?:years?\s*old)?', text, re.IGNORECASE):
            temporal.append({"type": "age", "subject": m.group(1).strip(), "value": int(m.group(2)), "text": m.group(0)})
        
        # --- Durations ("30 minutes", "2 hours", "3 days") ---
        # Require the unit word right after the number (no GPA-style false matches)
        for m in re.finditer(r'(\d{1,9})\s+(minute|hour|day|week|month|year)s?\b', text, re.IGNORECASE):
            temporal.append({"type": "duration", "value": int(m.group(1)), "unit": m.group(2).lower(), "text": m.group(0)})

        # --- Explicit quantities with labels ("12 tomato plants", "$15,000", "412 pages") ---
        for m in re.finditer(r'(\d{1,20}(?:,\d{3}){0,9}(?:\.\d{1,9})?)\s+(\w{1,40}(?:\s+\w{1,40}){0,2})', text):
            num_str = m.group(1).replace(',', '')
            label = m.group(2).strip()
            # Skip false matches like "3 hours" (already caught by duration)
            if any(u in label.lower() for u in ['plant', 'page', 'cost', 'price', '$', 'dollar', 'gpa', 'novel']):
                try:
                    temporal.append({"type": "quantity", "value": float(num_str), "label": label, "text": m.group(0)})
                except:
                    # An unparseable quantity is skipped so the rest of extraction continues.
                    pass

        return temporal

    def get_neighborhood(self, query: str, hops: int = 2, text_weight: float = 0.3,
                         top_seeds: int = 5) -> Dict[str, float]:
        """
        Find memories related to a query by hybrid BM25+embedding + graph traversal.
        
        1. Score all memories with embedding cosine + BM25 keyword
        2. Normalize each to [0,1], then fuse: (1-w) embedding + w BM25
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
        
        # --- Fuse: normalize → (1-w) embedding + w BM25 ---
        embed_weight = 1.0 - text_weight
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
                fused[mid] = embed_weight * e_norm + text_weight * b_norm
        
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
        
        # Apply learned edge weight boosts from self-learning
        self._apply_learned_boosts(reachable, seeds)
        
        return reachable

    def _apply_learned_boosts(self, reachable: Dict[str, float], seeds: List[str]):
        """Apply learned edge weight boosts to reachable nodes based on seed connections."""
        if not self.learned_weights:
            return
        seed_set = set(seeds)
        for nid in reachable:
            boost_sum = 0.0
            for seed in seed_set:
                key = (nid, seed) if nid < seed else (seed, nid)
                if key in self.learned_weights:
                    boost_sum += self.learned_weights[key]
            if boost_sum > 0:
                reachable[nid] *= (1.0 + min(boost_sum, 1.0) * 0.12)

    def recall_with_learning(self, query: str, k: int = 5, **kwargs) -> List[Dict]:
        """
        Recall memories and log the pattern for learning.
        Learning boosts are applied automatically via get_neighborhood.
        """
        hops = kwargs.get("hops", 2)
        top_seeds = kwargs.get("top_seeds", 5)
        neighborhood = self.get_neighborhood(query=query, hops=hops, top_seeds=top_seeds)
        
        sorted_nodes = sorted(neighborhood.items(), key=lambda x: x[1], reverse=True)[:k]
        results = [
            {"id": nid, "text": self.memories[nid].text, "score": score}
            for nid, score in sorted_nodes
        ]
        
        node_ids = [r["id"] for r in results]
        self.recall_tracker.log_recall(query, node_ids)
        
        return results

    def learn(self, min_co_recalls: int = 1):
        """
        Process recall history and update learned edge weights.
        Call this periodically (e.g., after every 100 recalls).
        """
        self.learned_weights = self.recall_tracker.learn_weights(min_co_recalls)
        return self.learned_weights

    def get_memory_scores(self) -> Dict[str, float]:
        """Get per-memory value scores based on recall frequency."""
        return self.recall_tracker.get_memory_values()

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

    def save_json(self, path: str):
        """Persist graph to disk as JSON (recoverable backup)."""
        import json
        data = {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "exported_at": __import__("datetime").datetime.utcnow().isoformat(),
            "memories": [
                {"id": mid, "text": mem.text, "source": getattr(mem, "source", "")}
                for mid, mem in self.memories.items()
            ],
            "semantic_schema_version": 1,
            "semantic_facts": [
                self.semantic_store.facts[fact_id].to_dict()
                for fact_id in sorted(self.semantic_store.facts)
            ],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str, model: str = None) -> 'MemoryGraph':
        """Load persisted graph from pickle."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        
        # Handle genuinely untyped old format: dict with G + memories.
        # A typed marker in this legacy branch must not be silently discarded.
        if isinstance(obj, dict) and 'G' in obj and 'memories' in obj:
            if any(key in obj for key in (
                    'semantic_store', 'semantic_facts', 'semantic_schema_version')):
                raise ValueError("legacy dict pickle contains typed state; refusing data loss")
            from .graph import MemoryGraph as MG
            g = MG(similarity_threshold=0.3, model=model)
            g.graph = obj['G']
            g.memories = obj['memories']
            g._rebuild_indices()
            return g
        
        # New format: serialized MemoryGraph
        if not isinstance(obj, MemoryGraph):
            raise ValueError("unsupported QMG pickle format")

        # Third-generation pickles carry vestigial attributes the strict pair
        # validator rejects; drop them under guard so this pair converges on
        # the next save instead of failing its own validator forever.
        _strip_legacy_graph_attrs(obj)

        if model:
            obj._model_name = model
            obj._embedder = None  # Force re-init on next use

        # Normalize all new-format memories even when cached indices exist.
        for mem in obj.memories.values():
            mem.timestamp = obj._normalize_timestamp(mem.timestamp)
        
        # Rebuild indices if missing (backward compat)
        if not hasattr(obj, '_entity_index') or not obj._entity_index:
            obj._rebuild_indices()
        if not hasattr(obj, '_bm25_corpus'):
            obj._bm25_corpus = []
        if not hasattr(obj, '_bm25'):
            obj._bm25 = None
        if not hasattr(obj, 'recall_tracker'):
            obj.recall_tracker = RecallTracker("")
        if not hasattr(obj, 'learned_weights'):
            obj.learned_weights = {}
        if not hasattr(obj, 'semantic_store'):
            obj.semantic_store = SemanticStore()
        elif not isinstance(obj.semantic_store, SemanticStore):
            raise ValueError("malformed semantic store; refusing to drop typed facts")
        else:
            # Full revalidation on load: canonical IDs, types, dates, evidence
            # shape, and source-memory linkage are all checked before indexes
            # are published; malformed state fails closed with ValueError.
            obj.semantic_store.rebuild_indices(memories=obj.memories)
        
        return obj

    def _rebuild_indices(self):
        """Rebuild entity index and BM25 corpus from existing memories."""
        self._entity_index = {}
        self._bm25_corpus = []
        for mid, mem in self.memories.items():
            # Old dict-format graphs are rebuilt here during load.
            mem.timestamp = self._normalize_timestamp(mem.timestamp)
            for entity in mem.entities:
                if entity not in self._entity_index:
                    self._entity_index[entity] = set()
                self._entity_index[entity].add(mid)
            self._bm25_corpus.append(mem.text)
        self._bm25 = None
