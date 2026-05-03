"""
Quantum Memory Graph API Server.

REST API for the graph + QAOA memory system.

Copyright 2026 Coinkong (Chef's Attraction). MIT License.
"""

import os
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from .graph import MemoryGraph
from .pipeline import store, store_batch, recall, get_graph, set_graph, get_stm
from .recency import ShortTermMemory

API_TOKEN = os.environ.get("QMG_API_TOKEN", "")
QMG_MODEL = os.environ.get("QMG_MODEL", None)  # e.g. thenlper/gte-large


async def verify_token(request: Request):
    if request.url.path in ("/", "/docs", "/openapi.json", "/redoc"):
        return
    if not API_TOKEN:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI(
    title="Quantum Memory Graph API",
    version="0.1.0",
    description="Knowledge graph + QAOA subgraph optimization for AI agent memory",
    dependencies=[Depends(verify_token)],
)

# Initialize graph on startup
_graph = None


@app.on_event("startup")
async def startup():
    global _graph
    threshold = float(os.environ.get("QMG_SIMILARITY_THRESHOLD", "0.3"))
    _graph = MemoryGraph(similarity_threshold=threshold, model=QMG_MODEL)
    set_graph(_graph)
    # Initialize short-term memory
    stm = get_stm()
    print(f"  Model: {_graph._model_name}")
    print(f"  Short-term memory: enabled (recency + working memory + conversation)")


@app.get("/")
async def health():
    g = get_graph()
    return {
        "status": "operational",
        "service": "Quantum Memory Graph",
        "version": "0.1.0",
        "graph": g.stats() if g and g.memories else {"nodes": 0, "edges": 0},
        "auth": "enabled" if API_TOKEN else "disabled",
    }


class StoreRequest(BaseModel):
    text: str
    entities: Optional[List[str]] = None
    source: str = ""


@app.post("/store")
async def api_store(req: StoreRequest):
    result = store(
        text=req.text,
        entities=req.entities,
        source=req.source,
    )
    return result


class StoreBatchRequest(BaseModel):
    texts: List[str]
    sources: Optional[List[str]] = None


@app.post("/store-batch")
async def api_store_batch(req: StoreBatchRequest):
    result = store_batch(
        texts=req.texts,
        sources=req.sources,
    )
    return result


class RecallRequest(BaseModel):
    query: str
    k: int = 5
    hops: int = 2
    top_seeds: int = 5
    alpha: float = 0.4
    beta_conn: float = 0.35
    gamma_cov: float = 0.25
    max_candidates: int = 14


@app.post("/recall")
async def api_recall(req: RecallRequest):
    result = recall(
        query=req.query,
        K=req.k,
        hops=req.hops,
        top_seeds=req.top_seeds,
        alpha=req.alpha,
        beta_conn=req.beta_conn,
        gamma_cov=req.gamma_cov,
        max_candidates=req.max_candidates,
    )
    return result


class QuantumRecallRequest(BaseModel):
    query: str
    user_id: str = "dustin"
    k: int = 5
    max_candidates: int = 14


@app.post("/quantum-recall")
async def api_quantum_recall(req: QuantumRecallRequest):
    """Compatibility endpoint for mem0-bridge plugin."""
    result = recall(
        query=req.query,
        K=req.k,
        max_candidates=req.max_candidates,
    )
    # Transform to mem0-bridge expected format
    memories = []
    for m in result.get("memories", []):
        memories.append({
            "memory": m["text"],
            "score": m.get("relevance", 0),
            "entities": m.get("entities", []),
            "connections": len(m.get("connections", [])),
        })
    return {
        "ok": True,
        "memories": memories,
        "method": result.get("method", "qaoa"),
        "graph_stats": result.get("graph_stats", {}),
    }


@app.get("/stats")
async def api_stats():
    g = get_graph()
    stm = get_stm()
    stats = g.stats() if g else {"nodes": 0, "edges": 0}
    stats["short_term_memory"] = stm.stats()
    return stats


def main():
    host = os.environ.get("QMG_HOST", "0.0.0.0")
    port = int(os.environ.get("QMG_PORT", "8502"))
    print(f"⚛️🧠 Quantum Memory Graph API starting on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
