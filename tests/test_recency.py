"""
Tests for short-term memory layer.

DK 🦍 — RED → GREEN → SHIP
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from quantum_memory_graph.recency import (
    RecencyBooster, WorkingMemory, ConversationContext, ShortTermMemory
)


class TestRecencyBooster:
    def test_recent_memory_gets_high_boost(self):
        booster = RecencyBooster()
        now = datetime.now()
        # 5 minutes ago
        recent = now - timedelta(minutes=5)
        boost = booster.boost(recent, now)
        assert boost > 0.2, f"Recent memory should get high boost, got {boost}"
    
    def test_old_memory_gets_no_boost(self):
        booster = RecencyBooster()
        now = datetime.now()
        old = now - timedelta(days=30)
        boost = booster.boost(old, now)
        assert boost == 0.0, f"Old memory should get no boost, got {boost}"
    
    def test_day_old_gets_moderate_boost(self):
        booster = RecencyBooster()
        now = datetime.now()
        yesterday = now - timedelta(hours=12)
        boost = booster.boost(yesterday, now)
        assert 0.05 < boost < 0.2, f"Day-old memory should get moderate boost, got {boost}"
    
    def test_week_old_gets_small_boost(self):
        booster = RecencyBooster()
        now = datetime.now()
        last_week = now - timedelta(days=5)
        boost = booster.boost(last_week, now)
        assert 0.0 < boost < 0.1, f"Week-old should get small boost, got {boost}"
    
    def test_apply_boosts_scores(self):
        booster = RecencyBooster()
        now = datetime.now()
        
        mem_recent = MagicMock()
        mem_recent.timestamp = now - timedelta(minutes=5)
        mem_old = MagicMock()
        mem_old.timestamp = now - timedelta(days=30)
        
        memories = {"m1": mem_recent, "m2": mem_old}
        scores = {"m1": 0.5, "m2": 0.5}
        
        boosted = booster.apply(scores, memories, now)
        assert boosted["m1"] > boosted["m2"], "Recent memory should score higher"
    
    def test_custom_decay_rates(self):
        custom = {1800: 0.5, 7200: 0.2}  # 30min, 2hr
        booster = RecencyBooster(decay_rates=custom)
        now = datetime.now()
        
        very_recent = now - timedelta(minutes=10)
        boost = booster.boost(very_recent, now)
        assert boost > 0.3, f"Custom rate should give high boost, got {boost}"


class TestWorkingMemory:
    def test_add_and_retrieve(self):
        wm = WorkingMemory(capacity=5)
        wm.add("m1", "Hello world")
        wm.add("m2", "Second memory")
        
        assert len(wm) == 2
        assert wm.contains("m1")
        assert wm.contains("m2")
    
    def test_capacity_limit(self):
        wm = WorkingMemory(capacity=3)
        wm.add("m1", "First")
        wm.add("m2", "Second")
        wm.add("m3", "Third")
        wm.add("m4", "Fourth")  # Should push out m1
        
        assert len(wm) == 3
        assert not wm.contains("m1"), "Oldest should be evicted"
        assert wm.contains("m4"), "Newest should be present"
    
    def test_no_duplicates(self):
        wm = WorkingMemory(capacity=5)
        wm.add("m1", "Hello")
        wm.add("m1", "Hello")  # Duplicate
        
        assert len(wm) == 1
    
    def test_get_recent_order(self):
        wm = WorkingMemory(capacity=5)
        wm.add("m1", "First")
        wm.add("m2", "Second")
        wm.add("m3", "Third")
        
        recent = wm.get_recent(2)
        assert recent[0]["id"] == "m3", "Most recent should be first"
        assert recent[1]["id"] == "m2"
    
    def test_clear(self):
        wm = WorkingMemory(capacity=5)
        wm.add("m1", "Hello")
        wm.clear()
        assert len(wm) == 0
        assert not wm.contains("m1")


class TestConversationContext:
    def test_add_turns_and_boost(self):
        ctx = ConversationContext(boost=0.25)
        ctx.add_turn("What's our tech stack?", memory_ids=["m1", "m2"])
        
        scores = {"m1": 0.5, "m2": 0.5, "m3": 0.5}
        boosted = ctx.apply_boost(scores)
        
        assert boosted["m1"] == 0.75, "Conversation memory should be boosted"
        assert boosted["m3"] == 0.5, "Non-conversation memory unchanged"
    
    def test_start_new_session_clears(self):
        ctx = ConversationContext()
        ctx.add_turn("Turn 1", memory_ids=["m1"])
        ctx.start_session("new_session")
        
        assert len(ctx) == 0
        assert len(ctx.get_context_ids()) == 0
    
    def test_window_limit(self):
        ctx = ConversationContext(window_size=3)
        ctx.add_turn("Turn 1")
        ctx.add_turn("Turn 2")
        ctx.add_turn("Turn 3")
        ctx.add_turn("Turn 4")  # Should push out Turn 1
        
        assert len(ctx) == 3


class TestShortTermMemory:
    def test_full_pipeline(self):
        stm = ShortTermMemory(
            working_memory_size=10,
            conversation_window=5,
            conversation_boost=0.2,
        )
        
        now = datetime.now()
        
        # Simulate storing memories
        stm.on_store("m1", "Recent memory", now - timedelta(minutes=5))
        stm.on_store("m2", "Old memory", now - timedelta(days=30))
        
        # Create mock memories
        mem1 = MagicMock()
        mem1.timestamp = now - timedelta(minutes=5)
        mem2 = MagicMock()
        mem2.timestamp = now - timedelta(days=30)
        
        memories = {"m1": mem1, "m2": mem2}
        scores = {"m1": 0.5, "m2": 0.5}
        
        boosted = stm.apply(scores, memories, now)
        assert boosted["m1"] > boosted["m2"], "Recent memory should rank higher"
    
    def test_working_memory_inclusion(self):
        stm = ShortTermMemory(working_memory_size=5)
        stm.on_store("m_working", "Just stored this")
        
        # Verify working memory has the item
        assert stm.working is not None
        assert stm.working.contains("m_working")
        
        # m_working not in search results but in working memory
        memories = {"m_working": MagicMock(timestamp=datetime.now())}
        scores = {"m_other": 0.5}  # m_working not in scores
        
        boosted = stm.apply(scores, memories)
        assert "m_working" in boosted, "Working memory should be injected"
    
    def test_disable_features(self):
        stm = ShortTermMemory(
            enable_recency=False,
            enable_working_memory=False,
            enable_conversation=False,
        )
        
        scores = {"m1": 0.5}
        result = stm.apply(scores, {})
        assert result == scores, "Disabled features should pass through"
    
    def test_stats(self):
        stm = ShortTermMemory(working_memory_size=10)
        stm.on_store("m1", "test")
        
        assert stm.working is not None
        assert len(stm.working) == 1
        
        stats = stm.stats()
        assert stats["working_memory_size"] == 1
        assert stats["recency_enabled"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
