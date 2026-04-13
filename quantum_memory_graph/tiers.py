"""
Semantic Memory Tiers — CPU-cache-style memory hierarchy.

Three tiers:
  Hot  (< 1 hour):  Session cache, injected every message
  Warm (1-24 hours): Today's events, injected on relevant queries  
  Cold (> 24 hours): Full mem0 store, recalled on demand

Each tier has different latency, injection strategy, and eviction policy.

DK 🦍 — v1.0.0
"""

import time
import sqlite3
import json
import os
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class TieredMemory:
    """A memory entry with tier metadata."""
    id: str
    text: str
    agent_id: str = ""
    timestamp: float = 0.0  # Unix timestamp
    source: str = ""
    entities: List[str] = field(default_factory=list)
    tier: str = "hot"  # hot, warm, cold
    access_count: int = 0
    last_accessed: float = 0.0


class HotTier:
    """
    Hot memory — last hour, always injected.
    
    In-memory deque. Zero latency. Auto-evicts after TTL.
    """
    
    def __init__(self, capacity: int = 50, ttl_seconds: int = 3600):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self._buffer: deque = deque(maxlen=capacity)
        self._ids: set = set()
    
    def add(self, memory: TieredMemory):
        if memory.id in self._ids:
            return
        memory.tier = "hot"
        memory.timestamp = memory.timestamp or time.time()
        self._buffer.append(memory)
        self._ids.add(memory.id)
        self._evict_expired()
    
    def get_all(self) -> List[TieredMemory]:
        """Get all hot memories (for injection into every prompt)."""
        self._evict_expired()
        return list(self._buffer)
    
    def _evict_expired(self):
        """Remove memories older than TTL."""
        now = time.time()
        while self._buffer and (now - self._buffer[0].timestamp) > self.ttl:
            old = self._buffer.popleft()
            self._ids.discard(old.id)
    
    def demote(self) -> List[TieredMemory]:
        """Get and remove memories ready to demote to warm tier."""
        now = time.time()
        demoted = []
        while self._buffer and (now - self._buffer[0].timestamp) > self.ttl:
            old = self._buffer.popleft()
            self._ids.discard(old.id)
            old.tier = "warm"
            demoted.append(old)
        return demoted
    
    def __len__(self):
        return len(self._buffer)


class WarmTier:
    """
    Warm memory — today's events (1-24 hours old).
    
    SQLite-backed for persistence. Injected when relevant to query.
    Survives process restarts.
    """
    
    def __init__(self, db_path: str = None, ttl_seconds: int = 86400):
        self.ttl = ttl_seconds
        self.db_path = db_path or os.path.expanduser("~/.qmg/warm_tier.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS warm_memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                agent_id TEXT DEFAULT '',
                timestamp REAL NOT NULL,
                source TEXT DEFAULT '',
                entities TEXT DEFAULT '[]',
                access_count INTEGER DEFAULT 0,
                last_accessed REAL DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_warm_ts ON warm_memories(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_warm_agent ON warm_memories(agent_id)")
        conn.commit()
        conn.close()
    
    def add(self, memory: TieredMemory):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO warm_memories 
                (id, text, agent_id, timestamp, source, entities, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id, memory.text, memory.agent_id,
                memory.timestamp or time.time(), memory.source,
                json.dumps(memory.entities), memory.access_count,
                memory.last_accessed,
            ))
            conn.commit()
        finally:
            conn.close()
    
    def add_batch(self, memories: List[TieredMemory]):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany("""
                INSERT OR REPLACE INTO warm_memories 
                (id, text, agent_id, timestamp, source, entities, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [(
                m.id, m.text, m.agent_id, m.timestamp or time.time(),
                m.source, json.dumps(m.entities), m.access_count, m.last_accessed,
            ) for m in memories])
            conn.commit()
        finally:
            conn.close()
    
    def search(self, query: str, agent_id: str = None, limit: int = 10) -> List[TieredMemory]:
        """
        Search warm tier by text content.
        Simple keyword search — fast, no embedding needed.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Evict expired first
            cutoff = time.time() - self.ttl
            conn.execute("DELETE FROM warm_memories WHERE timestamp < ?", (cutoff,))
            conn.commit()
            
            # Search with LIKE for key terms
            terms = [t.strip() for t in query.lower().split() if len(t.strip()) > 2]
            if not terms:
                return []
            
            conditions = ["LOWER(text) LIKE ?"] * len(terms)
            params = [f"%{t}%" for t in terms[:5]]  # Cap at 5 terms
            
            where = " OR ".join(conditions)
            if agent_id:
                where = f"({where}) AND agent_id = ?"
                params.append(agent_id)
            
            rows = conn.execute(f"""
                SELECT id, text, agent_id, timestamp, source, entities, access_count, last_accessed
                FROM warm_memories WHERE {where}
                ORDER BY timestamp DESC LIMIT ?
            """, params + [limit]).fetchall()
            
            return [self._row_to_memory(r) for r in rows]
        finally:
            conn.close()
    
    def get_recent(self, agent_id: str = None, limit: int = 20) -> List[TieredMemory]:
        """Get most recent warm memories."""
        conn = sqlite3.connect(self.db_path)
        try:
            if agent_id:
                rows = conn.execute("""
                    SELECT id, text, agent_id, timestamp, source, entities, access_count, last_accessed
                    FROM warm_memories WHERE agent_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (agent_id, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT id, text, agent_id, timestamp, source, entities, access_count, last_accessed
                    FROM warm_memories ORDER BY timestamp DESC LIMIT ?
                """, (limit,)).fetchall()
            return [self._row_to_memory(r) for r in rows]
        finally:
            conn.close()
    
    def demote_expired(self) -> List[TieredMemory]:
        """Remove and return memories older than TTL (for cold tier)."""
        cutoff = time.time() - self.ttl
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("""
                SELECT id, text, agent_id, timestamp, source, entities, access_count, last_accessed
                FROM warm_memories WHERE timestamp < ?
            """, (cutoff,)).fetchall()
            
            if rows:
                conn.execute("DELETE FROM warm_memories WHERE timestamp < ?", (cutoff,))
                conn.commit()
            
            memories = [self._row_to_memory(r) for r in rows]
            for m in memories:
                m.tier = "cold"
            return memories
        finally:
            conn.close()
    
    def count(self, agent_id: str = None) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            if agent_id:
                return conn.execute(
                    "SELECT COUNT(*) FROM warm_memories WHERE agent_id = ?", (agent_id,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM warm_memories").fetchone()[0]
        finally:
            conn.close()
    
    def _row_to_memory(self, row) -> TieredMemory:
        return TieredMemory(
            id=row[0], text=row[1], agent_id=row[2],
            timestamp=row[3], source=row[4],
            entities=json.loads(row[5]) if row[5] else [],
            access_count=row[6], last_accessed=row[7],
            tier="warm",
        )


class MemoryTierManager:
    """
    Unified tier manager — coordinates hot/warm/cold.
    
    Usage:
        tiers = MemoryTierManager(agent_id="donkeykong")
        
        # Store new memory
        tiers.store("User prefers dark mode", source="chat")
        
        # Get what to inject into prompt
        hot = tiers.get_hot()          # Always inject these
        warm = tiers.get_warm(query)    # Inject if relevant
        
        # Maintenance
        tiers.tick()  # Demote expired memories between tiers
    """
    
    def __init__(self, agent_id: str = "", db_path: str = None,
                 hot_capacity: int = 50, hot_ttl: int = 3600,
                 warm_ttl: int = 86400):
        self.agent_id = agent_id
        self.hot = HotTier(capacity=hot_capacity, ttl_seconds=hot_ttl)
        self.warm = WarmTier(db_path=db_path, ttl_seconds=warm_ttl)
    
    def store(self, text: str, entities: List[str] = None,
              source: str = "", memory_id: str = None) -> TieredMemory:
        """Store a new memory — goes to hot tier first."""
        import hashlib
        mid = memory_id or hashlib.md5(text.encode()).hexdigest()[:16]
        
        mem = TieredMemory(
            id=mid, text=text, agent_id=self.agent_id,
            timestamp=time.time(), source=source,
            entities=entities or [], tier="hot",
        )
        self.hot.add(mem)
        return mem
    
    def get_hot(self) -> List[TieredMemory]:
        """Get all hot memories (inject into every message)."""
        return self.hot.get_all()
    
    def get_warm(self, query: str = None, limit: int = 10) -> List[TieredMemory]:
        """Get relevant warm memories (inject when query matches)."""
        if query:
            return self.warm.search(query, agent_id=self.agent_id, limit=limit)
        return self.warm.get_recent(agent_id=self.agent_id, limit=limit)
    
    def tick(self):
        """
        Run tier maintenance — demote expired memories.
        Call this periodically (every minute or so).
        """
        # Hot → Warm
        demoted = self.hot.demote()
        if demoted:
            self.warm.add_batch(demoted)
        
        # Warm → Cold (just remove from warm; cold = mem0/QMG)
        self.warm.demote_expired()
    
    def stats(self) -> Dict:
        return {
            "hot_count": len(self.hot),
            "warm_count": self.warm.count(self.agent_id),
            "hot_ttl": self.hot.ttl,
            "warm_ttl": self.warm.ttl,
            "agent_id": self.agent_id,
        }
