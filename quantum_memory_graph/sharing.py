"""
Cross-Agent Memory Sharing — Controlled knowledge pools.

Two pools:
  Shared: Business rules, Chef's decisions, infrastructure facts
  Private: Agent-specific memories, personal context

Agents can READ shared pool, WRITE to shared pool (with tagging),
and NEVER read other agents' private pools (unless explicitly granted).

DK 🦍 — v1.0.0
"""

import time
import sqlite3
import json
import os
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class SharedMemory:
    """A memory in the shared pool."""
    id: str
    text: str
    author_agent: str  # Who wrote it
    timestamp: float
    category: str = "general"  # business, technical, rules, people, general
    entities: List[str] = field(default_factory=list)
    access_control: List[str] = field(default_factory=list)  # Empty = all agents
    read_count: int = 0


CATEGORIES = ["business", "technical", "rules", "people", "general"]


class SharedMemoryPool:
    """
    Shared knowledge pool for cross-agent access.
    
    All agents can:
      - Read from the shared pool
      - Write to the shared pool (tagged with their agent_id)
      - Search by category, keyword, or entity
    
    Access control:
      - access_control=[] → all agents can read
      - access_control=["daisy", "peach"] → only those agents
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.expanduser("~/.qmg/shared_pool.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                author_agent TEXT NOT NULL,
                timestamp REAL NOT NULL,
                category TEXT DEFAULT 'general',
                entities TEXT DEFAULT '[]',
                access_control TEXT DEFAULT '[]',
                read_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shared_cat ON shared_memories(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shared_author ON shared_memories(author_agent)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shared_ts ON shared_memories(timestamp)")
        conn.commit()
        conn.close()
    
    def store(self, text: str, author_agent: str, category: str = "general",
              entities: List[str] = None, access_control: List[str] = None,
              memory_id: str = None) -> SharedMemory:
        """Store a memory in the shared pool."""
        import hashlib
        mid = memory_id or hashlib.md5(f"{author_agent}:{text}".encode()).hexdigest()[:16]
        
        mem = SharedMemory(
            id=mid, text=text, author_agent=author_agent,
            timestamp=time.time(), category=category,
            entities=entities or [], access_control=access_control or [],
        )
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO shared_memories
                (id, text, author_agent, timestamp, category, entities, access_control, read_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mem.id, mem.text, mem.author_agent, mem.timestamp,
                mem.category, json.dumps(mem.entities),
                json.dumps(mem.access_control), mem.read_count,
            ))
            conn.commit()
        finally:
            conn.close()
        
        return mem
    
    def recall(self, query: str, requesting_agent: str,
               category: str = None, limit: int = 10) -> List[SharedMemory]:
        """
        Recall shared memories. Respects access control.
        
        Returns memories where:
          - access_control is empty (public), OR
          - requesting_agent is in access_control list
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Build query
            terms = [t.strip() for t in query.lower().split() if len(t.strip()) > 2]
            if not terms:
                # No search terms — return recent
                return self._get_recent(conn, requesting_agent, category, limit)
            
            conditions = []
            params = []
            
            # Text search (escape SQL wildcards)
            def _escape_like(s):
                return s.replace('%', '\\%').replace('_', '\\_')
            
            text_conds = ["LOWER(text) LIKE ? ESCAPE '\\'"] * min(len(terms), 5)
            text_params = [f"%{_escape_like(t)}%" for t in terms[:5]]
            conditions.append(f"({' OR '.join(text_conds)})")
            params.extend(text_params)
            
            # Category filter
            if category:
                conditions.append("category = ?")
                params.append(category)
            
            where = " AND ".join(conditions)
            
            rows = conn.execute(f"""
                SELECT id, text, author_agent, timestamp, category, 
                       entities, access_control, read_count
                FROM shared_memories WHERE {where}
                ORDER BY timestamp DESC LIMIT ?
            """, params + [limit * 2]).fetchall()  # Fetch extra, filter AC
            
            # Filter by access control
            results = []
            for row in rows:
                mem = self._row_to_memory(row)
                if self._can_access(mem, requesting_agent):
                    results.append(mem)
                    if len(results) >= limit:
                        break
            
            # Update read counts
            for mem in results:
                conn.execute(
                    "UPDATE shared_memories SET read_count = read_count + 1 WHERE id = ?",
                    (mem.id,)
                )
            conn.commit()
            
            return results
        finally:
            conn.close()
    
    def _get_recent(self, conn, requesting_agent: str,
                    category: str = None, limit: int = 10) -> List[SharedMemory]:
        if category:
            rows = conn.execute("""
                SELECT id, text, author_agent, timestamp, category,
                       entities, access_control, read_count
                FROM shared_memories WHERE category = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (category, limit * 2)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, text, author_agent, timestamp, category,
                       entities, access_control, read_count
                FROM shared_memories ORDER BY timestamp DESC LIMIT ?
            """, (limit * 2,)).fetchall()
        
        results = []
        for row in rows:
            mem = self._row_to_memory(row)
            if self._can_access(mem, requesting_agent):
                results.append(mem)
                if len(results) >= limit:
                    break
        return results
    
    def _can_access(self, mem: SharedMemory, agent_id: str) -> bool:
        """Check if agent can access this memory."""
        if not mem.access_control:
            return True  # Public
        return agent_id in mem.access_control
    
    def get_by_author(self, author_agent: str, limit: int = 50) -> List[SharedMemory]:
        """Get all shared memories written by a specific agent."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("""
                SELECT id, text, author_agent, timestamp, category,
                       entities, access_control, read_count
                FROM shared_memories WHERE author_agent = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (author_agent, limit)).fetchall()
            return [self._row_to_memory(r) for r in rows]
        finally:
            conn.close()
    
    def get_categories(self) -> Dict[str, int]:
        """Get memory counts per category."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("""
                SELECT category, COUNT(*) FROM shared_memories GROUP BY category
            """).fetchall()
            return {r[0]: r[1] for r in rows}
        finally:
            conn.close()
    
    def stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        try:
            total = conn.execute("SELECT COUNT(*) FROM shared_memories").fetchone()[0]
            authors = conn.execute(
                "SELECT author_agent, COUNT(*) FROM shared_memories GROUP BY author_agent"
            ).fetchall()
            categories = self.get_categories()
            return {
                "total_memories": total,
                "by_author": {r[0]: r[1] for r in authors},
                "by_category": categories,
            }
        finally:
            conn.close()
    
    def _row_to_memory(self, row) -> SharedMemory:
        return SharedMemory(
            id=row[0], text=row[1], author_agent=row[2],
            timestamp=row[3], category=row[4],
            entities=json.loads(row[5]) if row[5] else [],
            access_control=json.loads(row[6]) if row[6] else [],
            read_count=row[7],
        )
