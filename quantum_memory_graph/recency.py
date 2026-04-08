"""
Short-Term Memory Layer — Recency-aware retrieval for production agents.

Adds three capabilities:
1. Recency boost: Recent memories score higher (exponential decay)
2. Working memory buffer: Last N memories in hot cache for instant access
3. Conversation context: Current conversation gets priority

DK 🦍 — v0.4.0
"""

import math
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np


class RecencyBooster:
    """
    Apply time-based decay to memory scores.
    
    Recent memories get a boost that decays exponentially:
      - Last hour:  +0.30
      - Last day:   +0.15
      - Last week:  +0.05
      - Older:      +0.00
    
    Configurable via decay_rates dict.
    """
    
    DEFAULT_DECAY = {
        3600:    0.30,   # 1 hour
        86400:   0.15,   # 1 day
        604800:  0.05,   # 1 week
    }
    
    def __init__(self, decay_rates: Dict[int, float] = None,
                 max_boost: float = 0.3):
        self.decay_rates = decay_rates or self.DEFAULT_DECAY
        self.max_boost = max_boost
        # Sort by time window (ascending)
        self._sorted_rates = sorted(self.decay_rates.items())
    
    def boost(self, timestamp: datetime, now: datetime = None) -> float:
        """Calculate recency boost for a memory timestamp."""
        now = now or datetime.now()
        age_seconds = max(0, (now - timestamp).total_seconds())
        
        for window_seconds, boost_value in self._sorted_rates:
            if age_seconds <= window_seconds:
                # Smooth decay within the window
                ratio = age_seconds / window_seconds
                return boost_value * (1.0 - ratio * 0.5)  # Gradual falloff
        
        return 0.0
    
    def apply(self, scores: Dict[str, float], memories: Dict,
              now: datetime = None) -> Dict[str, float]:
        """
        Apply recency boost to a dict of {memory_id: score}.
        memories is {memory_id: Memory} for timestamp access.
        """
        now = now or datetime.now()
        boosted = {}
        for mid, score in scores.items():
            mem = memories.get(mid)
            if mem and hasattr(mem, 'timestamp') and mem.timestamp:
                boost = self.boost(mem.timestamp, now)
                boosted[mid] = score + boost
            else:
                boosted[mid] = score
        return boosted


class WorkingMemory:
    """
    Hot cache of the most recent N memories for instant access.
    
    - Stored in a deque (FIFO, O(1) append/pop)
    - Bypasses embedding search for very recent context
    - Always included in recall results if relevant
    """
    
    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._ids: set = set()
    
    def add(self, memory_id: str, text: str, timestamp: datetime = None):
        """Add a memory to the working buffer."""
        if memory_id in self._ids:
            return
        
        entry = {
            "id": memory_id,
            "text": text,
            "timestamp": timestamp or datetime.now(),
            "added_at": time.time(),
        }
        
        # If buffer is full, remove oldest from id set
        if len(self._buffer) >= self.capacity:
            oldest = self._buffer[0]
            self._ids.discard(oldest["id"])
        
        self._buffer.append(entry)
        self._ids.add(memory_id)
    
    def get_recent(self, n: int = None) -> List[Dict]:
        """Get the N most recent entries (newest first)."""
        entries = list(reversed(self._buffer))
        if n:
            entries = entries[:n]
        return entries
    
    def get_ids(self) -> set:
        """Get all memory IDs in working memory."""
        return self._ids.copy()
    
    def contains(self, memory_id: str) -> bool:
        return memory_id in self._ids
    
    def clear(self):
        self._buffer.clear()
        self._ids.clear()
    
    def __len__(self):
        return len(self._buffer)


class ConversationContext:
    """
    Track the current conversation for context-aware recall.
    
    Memories from the active conversation get a priority boost,
    simulating "what we were just talking about" awareness.
    """
    
    def __init__(self, window_size: int = 10, boost: float = 0.2):
        self.window_size = window_size
        self.boost = boost
        self._turns: deque = deque(maxlen=window_size)
        self._memory_ids: set = set()
        self._session_id: Optional[str] = None
    
    def start_session(self, session_id: str = None):
        """Start a new conversation session."""
        self._session_id = session_id or f"session_{int(time.time())}"
        self._turns.clear()
        self._memory_ids.clear()
    
    def add_turn(self, text: str, memory_ids: List[str] = None):
        """Add a conversation turn."""
        self._turns.append({
            "text": text,
            "timestamp": datetime.now(),
            "memory_ids": memory_ids or [],
        })
        if memory_ids:
            self._memory_ids.update(memory_ids)
    
    def get_context_ids(self) -> set:
        """Get memory IDs mentioned in current conversation."""
        return self._memory_ids.copy()
    
    def apply_boost(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Boost scores for memories in current conversation context."""
        boosted = {}
        for mid, score in scores.items():
            if mid in self._memory_ids:
                boosted[mid] = score + self.boost
            else:
                boosted[mid] = score
        return boosted
    
    @property
    def session_id(self):
        return self._session_id
    
    def __len__(self):
        return len(self._turns)


class ShortTermMemory:
    """
    Combined short-term memory layer.
    
    Wraps RecencyBooster + WorkingMemory + ConversationContext
    into a single interface that plugs into MemoryGraph.recall().
    
    Usage:
        stm = ShortTermMemory()
        
        # When storing
        stm.on_store(memory_id, text, timestamp)
        
        # When recalling
        boosted_scores = stm.apply(raw_scores, memories)
        
        # Conversation tracking
        stm.conversation.add_turn("What's our tech stack?")
    """
    
    def __init__(self,
                 recency_decay: Dict[int, float] = None,
                 working_memory_size: int = 20,
                 conversation_window: int = 10,
                 conversation_boost: float = 0.2,
                 enable_recency: bool = True,
                 enable_working_memory: bool = True,
                 enable_conversation: bool = True):
        
        self.recency = RecencyBooster(recency_decay) if enable_recency else None
        self.working = WorkingMemory(working_memory_size) if enable_working_memory else None
        self.conversation = ConversationContext(
            conversation_window, conversation_boost
        ) if enable_conversation else None
    
    def on_store(self, memory_id: str, text: str,
                 timestamp: datetime = None):
        """Called when a new memory is stored."""
        if self.working is not None:
            self.working.add(memory_id, text, timestamp)
    
    def apply(self, scores: Dict[str, float], memories: Dict,
              now: datetime = None) -> Dict[str, float]:
        """
        Apply all short-term memory boosts to raw scores.
        
        Order: recency → conversation context → working memory inclusion
        """
        result = dict(scores)
        
        # 1. Recency boost
        if self.recency is not None:
            result = self.recency.apply(result, memories, now)
        
        # 2. Conversation context boost
        if self.conversation is not None:
            result = self.conversation.apply_boost(result)
        
        # 3. Ensure working memory items are included
        if self.working is not None:
            for mid in self.working.get_ids():
                if mid not in result and mid in memories:
                    # Add working memory items with a base score
                    result[mid] = 0.1  # Low but present
        
        return result
    
    def get_working_memory_ids(self) -> set:
        """Get IDs in working memory buffer."""
        if self.working is not None:
            return self.working.get_ids()
        return set()
    
    def stats(self) -> Dict:
        return {
            "working_memory_size": len(self.working) if self.working else 0,
            "conversation_turns": len(self.conversation) if self.conversation else 0,
            "recency_enabled": self.recency is not None,
        }
