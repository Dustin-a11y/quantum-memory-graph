"""
Quantum Memory Graph API Server — v1.1.1

REST API for the full memory system:
  - Store/recall with graph + QAOA
  - Deduplication
  - Tiered memory (hot/warm/cold)
  - Cross-agent sharing
  - Obsidian export

Copyright 2026 Coinkong (Chef's Attraction). MIT License.
"""

import os
import hmac
import time as _time
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn

from .graph import MemoryGraph
from .pipeline import store, store_batch, recall, get_graph, set_graph, get_stm
from .recency import ShortTermMemory
from .dedup import MemoryDeduplicator, deduplicate
from .tiers import MemoryTierManager
from .sharing import SharedMemoryPool
from .obsidian import export_vault, export_from_mem0

API_TOKEN = os.environ.get("QMG_API_TOKEN", "")
QMG_MODEL = os.environ.get("QMG_MODEL", None)

# Shared instances
_tiers: dict = {}  # agent_id -> MemoryTierManager
_shared_pool: Optional[SharedMemoryPool] = None


async def verify_token(request: Request):
    if request.url.path in ("/", "/docs", "/openapi.json", "/redoc"):
        return
    if not API_TOKEN:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or not hmac.compare_digest(auth[7:], API_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI(
    title="Quantum Memory Graph API",
    version="1.0.0",
    description="Knowledge graph + QAOA + tiered memory + sharing for AI agents",
    dependencies=[Depends(verify_token)],
)


@app.on_event("startup")
async def startup():
    global _shared_pool
    threshold = float(os.environ.get("QMG_SIMILARITY_THRESHOLD", "0.3"))
    _graph = MemoryGraph(similarity_threshold=threshold, model=QMG_MODEL)
    set_graph(_graph)
    get_stm()
    _shared_pool = SharedMemoryPool()
    print(f"  Model: {_graph._model_name}")
    print(f"  Short-term memory: enabled")
    print(f"  Shared pool: enabled")
    print(f"  Dedup: available")
    print(f"  Tiers: available")
    print(f"  Obsidian export: available")


def _get_tiers(agent_id: str) -> MemoryTierManager:
    if agent_id not in _tiers:
        _tiers[agent_id] = MemoryTierManager(agent_id=agent_id)
    return _tiers[agent_id]


# ─── Health ───

@app.get("/")
async def health():
    g = get_graph()
    return {
        "status": "operational",
        "service": "Quantum Memory Graph",
        "version": "1.0.0",
        "features": ["graph", "qaoa", "stm", "dedup", "tiers", "sharing", "obsidian"],
        "graph": g.stats() if g and g.memories else {"nodes": 0, "edges": 0},
        "auth": "enabled" if API_TOKEN else "disabled",
    }


# ─── Store/Recall ───

class StoreRequest(BaseModel):
    text: str
    entities: Optional[List[str]] = None
    source: str = ""
    agent_id: str = ""


@app.post("/store")
async def api_store(req: StoreRequest):
    result = store(text=req.text, entities=req.entities, source=req.source)
    # Also store in hot tier if agent specified
    if req.agent_id:
        tiers = _get_tiers(req.agent_id)
        tiers.store(req.text, entities=req.entities, source=req.source)
    return result


class StoreBatchRequest(BaseModel):
    texts: List[str] = Field(..., max_length=500)  # Cap batch size
    sources: Optional[List[str]] = None


@app.post("/store-batch")
async def api_store_batch(req: StoreBatchRequest):
    return store_batch(texts=req.texts, sources=req.sources)


class RecallRequest(BaseModel):
    query: str
    k: int = 5
    hops: int = 2
    top_seeds: int = 5
    alpha: float = 0.4
    beta_conn: float = 0.35
    gamma_cov: float = 0.25
    max_candidates: int = 14
    agent_id: str = ""
    include_warm: bool = True


@app.post("/recall")
async def api_recall(req: RecallRequest):
    result = recall(
        query=req.query, K=req.k, hops=req.hops,
        top_seeds=req.top_seeds, alpha=req.alpha,
        beta_conn=req.beta_conn, gamma_cov=req.gamma_cov,
        max_candidates=req.max_candidates,
    )
    # Include warm tier memories if requested
    if req.agent_id and req.include_warm:
        tiers = _get_tiers(req.agent_id)
        warm = tiers.get_warm(req.query, limit=5)
        if warm:
            result["warm_memories"] = [
                {"text": m.text, "tier": "warm", "age_seconds": int(
                    _time.time() - m.timestamp
                )} for m in warm
            ]
    return result


class QuantumRecallRequest(BaseModel):
    query: str
    user_id: str = "dustin"
    k: int = 5
    max_candidates: int = 14


@app.post("/quantum-recall")
async def api_quantum_recall(req: QuantumRecallRequest):
    """Compatibility endpoint for mem0-bridge plugin."""
    result = recall(query=req.query, K=req.k, max_candidates=req.max_candidates)
    memories = []
    for m in result.get("memories", []):
        memories.append({
            "memory": m["text"],
            "score": m.get("relevance", 0),
            "entities": m.get("entities", []),
            "connections": len(m.get("connections", [])),
        })
    return {
        "ok": True, "memories": memories,
        "method": result.get("method", "qaoa"),
        "graph_stats": result.get("graph_stats", {}),
    }


# ─── Deduplication ───

class DedupRequest(BaseModel):
    threshold: float = 0.95
    dry_run: bool = False


@app.post("/dedup")
async def api_dedup(req: DedupRequest):
    """Find and merge duplicate memories."""
    return deduplicate(threshold=req.threshold, dry_run=req.dry_run)


# ─── Tiered Memory ───

class TierStoreRequest(BaseModel):
    text: str
    agent_id: str
    entities: Optional[List[str]] = None
    source: str = ""


@app.post("/tiers/store")
async def api_tier_store(req: TierStoreRequest):
    tiers = _get_tiers(req.agent_id)
    mem = tiers.store(req.text, entities=req.entities, source=req.source)
    return {"ok": True, "id": mem.id, "tier": mem.tier}


class TierRecallRequest(BaseModel):
    agent_id: str
    query: str = ""
    limit: int = 10


@app.post("/tiers/recall")
async def api_tier_recall(req: TierRecallRequest):
    tiers = _get_tiers(req.agent_id)
    hot = tiers.get_hot()
    warm = tiers.get_warm(req.query, limit=req.limit) if req.query else tiers.get_warm(limit=req.limit)
    return {
        "hot": [{"id": m.id, "text": m.text, "tier": "hot"} for m in hot],
        "warm": [{"id": m.id, "text": m.text, "tier": "warm"} for m in warm],
        "stats": tiers.stats(),
    }


@app.post("/tiers/tick")
async def api_tier_tick():
    """Run maintenance on all tier managers."""
    for agent_id, tiers in _tiers.items():
        tiers.tick()
    return {"ok": True, "agents_maintained": list(_tiers.keys())}


# ─── Cross-Agent Sharing ───

class ShareStoreRequest(BaseModel):
    text: str
    author_agent: str
    category: str = "general"
    entities: Optional[List[str]] = None
    access_control: Optional[List[str]] = None


@app.post("/shared/store")
async def api_shared_store(req: ShareStoreRequest):
    mem = _shared_pool.store(
        text=req.text, author_agent=req.author_agent,
        category=req.category, entities=req.entities,
        access_control=req.access_control,
    )
    return {"ok": True, "id": mem.id, "category": mem.category}


class ShareRecallRequest(BaseModel):
    query: str
    requesting_agent: str
    category: Optional[str] = None
    limit: int = 10


@app.post("/shared/recall")
async def api_shared_recall(req: ShareRecallRequest):
    results = _shared_pool.recall(
        query=req.query, requesting_agent=req.requesting_agent,
        category=req.category, limit=req.limit,
    )
    return {
        "memories": [
            {"id": m.id, "text": m.text, "author": m.author_agent,
             "category": m.category, "entities": m.entities}
            for m in results
        ],
        "count": len(results),
    }


@app.get("/shared/stats")
async def api_shared_stats():
    return _shared_pool.stats()


# ─── Obsidian Export ───

class ObsidianExportRequest(BaseModel):
    vault_path: str = Field(..., max_length=500)
    mem0_url: Optional[str] = Field(None, max_length=200)
    mem0_token: Optional[str] = Field(None, max_length=200)
    agents: Optional[List[str]] = Field(None, max_length=20)


@app.post("/obsidian/export")
async def api_obsidian_export(req: ObsidianExportRequest):
    """Export memories to an Obsidian vault."""
    if req.mem0_url:
        return export_from_mem0(
            mem0_url=req.mem0_url, vault_path=req.vault_path,
            api_token=req.mem0_token or "", agents=req.agents,
        )
    else:
        g = get_graph()
        return export_vault(g, req.vault_path)


# ─── Stats ───

@app.get("/stats")
async def api_stats():
    g = get_graph()
    stm = get_stm()
    stats = g.stats() if g else {"nodes": 0, "edges": 0}
    stats["short_term_memory"] = stm.stats()
    stats["tiers"] = {aid: t.stats() for aid, t in _tiers.items()}
    stats["shared_pool"] = _shared_pool.stats() if _shared_pool else {}
    return stats


def main():
    host = os.environ.get("QMG_HOST", "0.0.0.0")
    port = int(os.environ.get("QMG_PORT", "8502"))
    print(f"⚛️🧠 Quantum Memory Graph v1.1.1 starting on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
