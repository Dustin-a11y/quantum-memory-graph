"""
Tests for cross-agent memory sharing.
DK 🦍 — RED → GREEN → SHIP
"""

import pytest
import os
import tempfile

from quantum_memory_graph.sharing import SharedMemoryPool, SharedMemory


class TestSharedMemoryPool:
    def _make_pool(self):
        f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = f.name
        f.close()
        return SharedMemoryPool(db_path=db_path), db_path
    
    def test_store_and_recall(self):
        pool, db_path = self._make_pool()
        try:
            pool.store("CookUnity requires 72hr lead time",
                       author_agent="toadstool", category="business")
            
            results = pool.recall("CookUnity lead time", requesting_agent="daisy")
            assert len(results) >= 1
            assert "CookUnity" in results[0].text
        finally:
            os.unlink(db_path)
    
    def test_access_control_blocks(self):
        pool, db_path = self._make_pool()
        try:
            pool.store("Secret DK finding",
                       author_agent="donkeykong", category="technical",
                       access_control=["donkeykong", "luigi"])
            
            # DK can access
            dk_results = pool.recall("DK finding", requesting_agent="donkeykong")
            assert len(dk_results) >= 1
            
            # Daisy cannot access
            daisy_results = pool.recall("DK finding", requesting_agent="daisy")
            assert len(daisy_results) == 0
        finally:
            os.unlink(db_path)
    
    def test_public_access(self):
        pool, db_path = self._make_pool()
        try:
            pool.store("Chef wants weekly reports",
                       author_agent="daisy", category="rules")
            
            # All agents can read public memories
            for agent in ["donkeykong", "mario", "bowser"]:
                results = pool.recall("weekly reports", requesting_agent=agent)
                assert len(results) >= 1, f"{agent} should access public memory"
        finally:
            os.unlink(db_path)
    
    def test_category_filter(self):
        pool, db_path = self._make_pool()
        try:
            pool.store("Server runs Ubuntu", author_agent="dk", category="technical")
            pool.store("Chef wants automation", author_agent="daisy", category="business")
            
            tech = pool.recall("server", requesting_agent="dk", category="technical")
            biz = pool.recall("automation", requesting_agent="dk", category="business")
            
            assert len(tech) >= 1
            assert tech[0].category == "technical"
            assert len(biz) >= 1
            assert biz[0].category == "business"
        finally:
            os.unlink(db_path)
    
    def test_get_by_author(self):
        pool, db_path = self._make_pool()
        try:
            pool.store("DK memory 1", author_agent="donkeykong")
            pool.store("DK memory 2", author_agent="donkeykong")
            pool.store("Luigi memory", author_agent="luigi")
            
            dk_mems = pool.get_by_author("donkeykong")
            assert len(dk_mems) == 2
        finally:
            os.unlink(db_path)
    
    def test_stats(self):
        pool, db_path = self._make_pool()
        try:
            pool.store("Mem 1", author_agent="dk", category="technical")
            pool.store("Mem 2", author_agent="daisy", category="business")
            
            stats = pool.stats()
            assert stats["total_memories"] == 2
            assert "dk" in stats["by_author"]
            assert "technical" in stats["by_category"]
        finally:
            os.unlink(db_path)
    
    def test_get_categories(self):
        pool, db_path = self._make_pool()
        try:
            pool.store("A", author_agent="dk", category="technical")
            pool.store("B", author_agent="dk", category="technical")
            pool.store("C", author_agent="dk", category="business")
            
            cats = pool.get_categories()
            assert cats["technical"] == 2
            assert cats["business"] == 1
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
