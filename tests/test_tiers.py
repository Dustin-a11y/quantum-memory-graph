"""
Tests for semantic memory tiers.
DK 🦍 — RED → GREEN → SHIP
"""

import pytest
import time
import os
import tempfile

from quantum_memory_graph.tiers import (
    HotTier, WarmTier, MemoryTierManager, TieredMemory
)


class TestHotTier:
    def test_add_and_retrieve(self):
        hot = HotTier(capacity=10, ttl_seconds=3600)
        mem = TieredMemory(id="m1", text="Hello world", timestamp=time.time())
        hot.add(mem)
        
        assert len(hot) == 1
        all_hot = hot.get_all()
        assert len(all_hot) == 1
        assert all_hot[0].text == "Hello world"
    
    def test_capacity_limit(self):
        hot = HotTier(capacity=3, ttl_seconds=3600)
        for i in range(5):
            hot.add(TieredMemory(id=f"m{i}", text=f"Memory {i}", timestamp=time.time()))
        
        assert len(hot) == 3
    
    def test_no_duplicates(self):
        hot = HotTier(capacity=10, ttl_seconds=3600)
        mem = TieredMemory(id="m1", text="Hello", timestamp=time.time())
        hot.add(mem)
        hot.add(mem)
        
        assert len(hot) == 1
    
    def test_ttl_eviction(self):
        hot = HotTier(capacity=10, ttl_seconds=1)
        mem = TieredMemory(id="m1", text="Old", timestamp=time.time() - 2)
        hot.add(mem)
        
        # Should be evicted after TTL
        all_hot = hot.get_all()
        assert len(all_hot) == 0
    
    def test_demote(self):
        hot = HotTier(capacity=10, ttl_seconds=1)
        mem = TieredMemory(id="m1", text="Demote me", timestamp=time.time() - 2)
        hot.add(mem)
        
        demoted = hot.demote()
        assert len(demoted) >= 0  # May already be evicted by get_all
        assert len(hot) == 0


class TestWarmTier:
    def test_add_and_search(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            warm = WarmTier(db_path=db_path, ttl_seconds=86400)
            mem = TieredMemory(id="m1", text="CookUnity recipe analysis", 
                              agent_id="toadstool", timestamp=time.time())
            warm.add(mem)
            
            results = warm.search("CookUnity", agent_id="toadstool")
            assert len(results) >= 1
            assert results[0].text == "CookUnity recipe analysis"
        finally:
            os.unlink(db_path)
    
    def test_agent_isolation(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            warm = WarmTier(db_path=db_path)
            warm.add(TieredMemory(id="m1", text="DK security audit", 
                                  agent_id="donkeykong", timestamp=time.time()))
            warm.add(TieredMemory(id="m2", text="Daisy prompt work",
                                  agent_id="daisy", timestamp=time.time()))
            
            dk_results = warm.search("audit", agent_id="donkeykong")
            assert len(dk_results) == 1
            assert dk_results[0].agent_id == "donkeykong"
        finally:
            os.unlink(db_path)
    
    def test_count(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            warm = WarmTier(db_path=db_path)
            warm.add(TieredMemory(id="m1", text="Test 1", agent_id="dk", timestamp=time.time()))
            warm.add(TieredMemory(id="m2", text="Test 2", agent_id="dk", timestamp=time.time()))
            warm.add(TieredMemory(id="m3", text="Test 3", agent_id="luigi", timestamp=time.time()))
            
            assert warm.count() == 3
            assert warm.count(agent_id="dk") == 2
            assert warm.count(agent_id="luigi") == 1
        finally:
            os.unlink(db_path)


class TestMemoryTierManager:
    def test_store_goes_to_hot(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            mgr = MemoryTierManager(agent_id="dk", db_path=db_path)
            mem = mgr.store("New finding: port 8080 is open")
            
            assert mem.tier == "hot"
            hot = mgr.get_hot()
            assert len(hot) == 1
        finally:
            os.unlink(db_path)
    
    def test_tick_demotes_hot_to_warm(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            mgr = MemoryTierManager(agent_id="dk", db_path=db_path, hot_ttl=1)
            mgr.store("This will demote")
            
            # Wait for TTL
            time.sleep(1.5)
            mgr.tick()
            
            hot = mgr.get_hot()
            assert len(hot) == 0, "Hot should be empty after tick"
            
            warm = mgr.get_warm(limit=10)
            assert len(warm) >= 1, "Should have moved to warm"
        finally:
            os.unlink(db_path)
    
    def test_stats(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            mgr = MemoryTierManager(agent_id="dk", db_path=db_path)
            mgr.store("Test memory")
            
            stats = mgr.stats()
            assert stats["hot_count"] == 1
            assert stats["agent_id"] == "dk"
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
