"""
Quantum Memory Graph API Server.

REST API for the graph + QAOA memory system.

Copyright 2026 Coinkong (Chef's Attraction). MIT License.
"""

import asyncio
import ipaddress
import os
import pickle
import json
import hashlib
import logging
import math
import re
import tempfile
import threading
import numpy as np
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator
from typing import Any, Optional, List, Dict
import uvicorn

from . import rollback as _rollback
from . import semantic as _semantic
from .admission import AdmissionGate, graph_inventory, sha256 as _canonical_sha256
from .cache_attestation import verify_and_detach_embedder
from .graph import MemoryGraph, Memory
from .pipeline import store, store_batch, store_chunked, recall, recall_facts, answer, temporal_answer, get_graph, set_graph, get_stm, set_stm
from .recency import ShortTermMemory
from .search import hybrid_search

logger = logging.getLogger(__name__)

API_TOKEN = os.environ.get("QMG_API_TOKEN", "")

# Simple in-memory rate limiting
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        # Clean old requests
        self.requests[key] = [t for t in self.requests[key] if now - t < self.window_seconds]
        
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        self.requests[key].append(now)
        return True

# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

# Simple request counter for basic DoS protection
import threading
_store_counter = 0
_store_lock = threading.Lock()
_store_window_start = time.time()

def check_rate_limit():
    global _store_counter, _store_window_start
    with _store_lock:
        now = time.time()
        if now - _store_window_start > 60:
            _store_counter = 0
            _store_window_start = now
        _store_counter += 1
        if _store_counter > 100:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

QMG_MODEL = os.environ.get("QMG_MODEL", None)  # e.g. thenlper/gte-large


def _semantic_canary_enabled() -> bool:
    """Both explicit DK opt-in and a matching nonblank service token are required."""
    module_token = API_TOKEN if isinstance(API_TOKEN, str) else ""
    configured = os.environ.get("QMG_API_TOKEN", "")
    return (os.environ.get("QMG_SEMANTIC_CANARY") == "dk"
            and bool(module_token.strip()) and bool(configured.strip())
            and configured == module_token)


_ADMIN_PATHS = frozenset({
    "/semantic/admin/quiesce", "/semantic/admin/receipt",
    "/semantic/admin/resume", "/semantic/admin/status",
})
_SEMANTIC_PATHS = frozenset({"/semantic/store", "/semantic/recall"})


async def verify_token(request: Request):
    if request.url.path in (_SEMANTIC_PATHS | _ADMIN_PATHS) and not _semantic_canary_enabled():
        # Hide disabled DK-only routes before the ordinary Bearer challenge.
        raise HTTPException(status_code=404)
    if request.url.path in ("/", "/docs", "/openapi.json", "/redoc", "/stats"):
        return
    if not API_TOKEN:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def require_semantic_canary(request: Request):
    """Fail closed with 404 unless the opt-in typed canary is fully enabled.

    Evaluated per request (never cached at import) so env changes made after
    import are honoured. A disabled POST returns 404 regardless of Bearer
    status and stays unreachable even when the global token check is off.
    """
    if not _semantic_canary_enabled():
        raise HTTPException(status_code=404)

    # Pydantic's default 422 detail echoes rejected input. A lone JSON
    # surrogate in that echo makes Starlette's UTF-8 JSONResponse itself fail
    # with 500. Check every JSON string before model validation and return a
    # bounded error without reflecting any untrusted value. Legacy routes are
    # unaffected; this dependency runs only after the existing Bearer check.
    try:
        values = [await request.json()]
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(status_code=422, detail="Malformed semantic JSON") from exc
    while values:
        value = values.pop()
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise HTTPException(status_code=422, detail="Invalid semantic Unicode") from exc
        elif isinstance(value, float) and not math.isfinite(value):
            raise HTTPException(status_code=422, detail="Invalid semantic JSON number")
        elif isinstance(value, dict):
            values.extend(value.keys())
            values.extend(value.values())
        elif isinstance(value, list):
            values.extend(value)


async def require_admin_canary(request: Request):
    """Administrative controls require the DK opt-in, Bearer and socket loopback."""
    if not _semantic_canary_enabled():
        raise HTTPException(status_code=404)
    try:
        peer = ipaddress.ip_address(request.client.host if request.client else "")
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Loopback admin only") from exc
    if not peer.is_loopback:
        raise HTTPException(status_code=403, detail="Loopback admin only")


app = FastAPI(
    title="Quantum Memory Graph API",
    version="0.1.0",
    description="Knowledge graph + QAOA subgraph optimization for AI agent memory",
    dependencies=[Depends(verify_token)],
)

# Initialize graph on startup. This RLock serializes every paired persistence;
# AdmissionGate additionally counts ALL HTTP mutation routes before validation.
_graph = None
_PERSIST_LOCK = threading.RLock()
_ADMISSION = AdmissionGate()


@app.middleware("http")
async def _dk_ingress_barrier(request: Request, call_next):
    """Admit a whole mutating HTTP request or reject it; never queue a 503.

    Gate all present/future non-read HTTP methods rather than a manually kept
    route allowlist. Admin controls are authenticated separately and remain
    available while closed. A pre-existing admitted request is counted through
    handler completion, including recall/answer tracker side effects.
    """
    if (request.method.upper() in ("GET", "HEAD", "OPTIONS")
            or request.url.path in _ADMIN_PATHS):
        return await call_next(request)
    canary = _semantic_canary_enabled()
    if not canary and not _ADMISSION.is_paused():
        return await call_next(request)  # exact legacy behavior without opt-in
    if canary and request.headers.get("Authorization", "") != f"Bearer {API_TOKEN}":
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    if not _ADMISSION.enter():
        return JSONResponse(
            status_code=503,
            content={"detail": "DK ingress paused; write rejected (not queued)"},
        )
    try:
        return await call_next(request)
    finally:
        _ADMISSION.leave()


def _atomic_save(save_fn, destination: str):
    """Write beside the destination, then atomically publish the snapshot."""
    directory = os.path.dirname(destination) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(destination)}.", dir=directory)
    os.close(fd)
    try:
        save_fn(temporary)
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _persist_current_graph(*, background: bool = False):
    """Persist one graph generation while excluding reset and every other save.

    When paused, unsolicited autosave/shutdown must not rewrite a verified
    receipt or heal a corrupt pair before an operator sees the failure. A
    previously admitted HTTP handler may still persist while draining.
    """
    data_dir = os.environ.get("QMG_DATA_DIR", "")
    if not data_dir:
        return None
    with _PERSIST_LOCK:
        if background and _ADMISSION.is_paused():
            return None
        graph = get_graph()
        _atomic_save(graph.save, os.path.join(data_dir, "graph.pkl"))
        _atomic_save(graph.save_json, os.path.join(data_dir, "graph.json"))
        return graph


@app.on_event("startup")
async def startup():
    global _graph
    threshold = float(os.environ.get("QMG_SIMILARITY_THRESHOLD", "0.3"))
    _ADMISSION.arm_start_paused()
    start_paused = os.environ.get("QMG_SEMANTIC_START_PAUSED") == "1"
    if start_paused and not _semantic_canary_enabled():
        raise RuntimeError("start-paused DK canary requires the DK flag and nonblank token")

    # On a start-paused canary, a partial/corrupt pre-existing pair must never
    # be loaded as a clean graph and overwritten on the next autosave.
    data_dir = os.environ.get("QMG_DATA_DIR", "")
    loaded = False
    if start_paused:
        if not data_dir or os.path.islink(data_dir) or not os.path.isdir(data_dir):
            raise RuntimeError("start-paused DK canary requires a regular data directory")
        paths = [os.path.join(data_dir, name) for name in ("graph.pkl", "graph.json")]
        if any(os.path.lexists(path) for path in paths):
            try:
                _graph, _doc, _pkl, _json, _digests = _rollback._load_pair_checked(
                    data_dir, "startup pair", model=QMG_MODEL)
            except Exception as exc:
                raise RuntimeError("start-paused persisted pair is unverifiable") from exc
            set_graph(_graph)
            print(f"  Loaded {_graph.stats()['nodes']} memories from checked DK pair")
            loaded = True
    elif data_dir:
        pkl_path = os.path.join(data_dir, "graph.pkl")
        if os.path.exists(pkl_path):
            try:
                if QMG_MODEL is not None:
                    # The loader rewrites the model ID and drops an embedder
                    # cache under an override. Inspect both exact pair bytes
                    # and raw persisted state before allowing that rewrite.
                    _graph = _rollback._load_pair_checked(
                        data_dir, "startup pair", model=QMG_MODEL)[0]
                else:
                    _graph = MemoryGraph.load(pkl_path, model=QMG_MODEL)
                set_graph(_graph)
                stats = _graph.stats()
                print(f"  Loaded {stats['nodes']} memories from disk ({pkl_path})")
                loaded = True
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load persisted graph from {pkl_path}; refusing to start "
                    f"with an empty graph that could overwrite the existing snapshot"
                ) from e

    # Fall back to empty graph if nothing loaded
    if not loaded:
        _graph = MemoryGraph(similarity_threshold=threshold, model=QMG_MODEL)
        set_graph(_graph)

    # Initialize short-term memory
    stm = get_stm()
    print(f"  Model: {_graph._model_name}")
    print(f"  Short-term memory: enabled (recency + working memory + conversation)")


@app.on_event("shutdown")
async def shutdown():
    try:
        g = _persist_current_graph(background=True)
        if g is not None:
            print(f"  Saved {g.stats()['nodes']} memories to disk")
    except Exception as e:
        print(f"  WARNING: Failed to save graph: {e}")


# Periodic auto-save - writes graph.pkl every 5 minutes so data survives crashes
_SAVE_INTERVAL = 300  # 5 minutes

def _auto_save_loop():
    while True:
        threading.Event().wait(_SAVE_INTERVAL)
        try:
            g = _persist_current_graph(background=True)
            if g is not None:
                print(f"  [auto-save] Saved {g.stats()['nodes']} memories")
        except Exception as e:
            print(f"  [auto-save] WARNING: {e}")

threading.Thread(target=_auto_save_loop, daemon=True).start()


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
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = None


class SemanticStoreRequest(BaseModel):
    """Strict opt-in typed write: exactly text, source and explicit facts.

    Unknown top-level and unknown fact keys are rejected (extra='forbid' /
    SemanticStore.prepare), so callers can never smuggle ``id``, ``source``,
    ``source_memory_id`` or ``evidence_memory_ids`` into the write path.
    """

    model_config = ConfigDict(extra="forbid")

    text: StrictStr
    source: StrictStr
    semantic_facts: List[Dict[str, Any]] = Field(min_length=1)

    @field_validator("text", "source")
    @classmethod
    def _must_be_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be nonblank text")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("must be valid UTF-8 text") from exc
        return value


class SemanticRecallRequest(BaseModel):
    """Strict opt-in typed read; carries no winner and no mutation."""

    model_config = ConfigDict(extra="forbid")

    subject_id: StrictStr
    relation_type: StrictStr
    observed_as_of: Optional[StrictStr] = None
    valid_at: Optional[StrictStr] = None


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse common API and benchmark timestamps without guessing."""
    if not value:
        return None
    cleaned = value.strip()
    try:
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    raise HTTPException(status_code=422, detail=f"Unsupported timestamp format: {value}")


@app.post("/store")
async def api_store(req: StoreRequest):

    timestamp = parse_timestamp(req.timestamp)
    use_chunked = os.environ.get("QMG_CHUNKED_STORE", "").lower() in ("1", "true", "yes")
    if use_chunked and len(req.text) > 500:
        result = store_chunked(
            text=req.text,
            chunk_size=int(os.environ.get("QMG_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.environ.get("QMG_CHUNK_OVERLAP", "100")),
            entities=req.entities,
            timestamp=timestamp,
            source=req.source,
            metadata=req.metadata,
        )
    else:
        result = store(
            text=req.text,
            entities=req.entities,
            timestamp=timestamp,
            source=req.source,
            metadata=req.metadata,
        )
    return result


@app.post("/semantic/store", dependencies=[Depends(require_semantic_canary)])
async def api_semantic_store(req: SemanticStoreRequest):
    """Opt-in typed write: validate fully, then mutate exactly once.

    Canonical fact IDs are derived with ``SemanticStore.prepare`` *before* any
    mutation; ``pipeline.store`` re-validates and commits. Only validated
    caller errors become 422 — internal failures are not swallowed here.
    """
    try:
        source_memory_id = Memory.make_id(req.text)
        prepared = _semantic.SemanticStore.prepare(
            req.semantic_facts, source=req.source, source_memory_id=source_memory_id
        )
    except (ValueError, TypeError, KeyError) as exc:
        logger.exception("semantic store rejected malformed fact payload")
        raise HTTPException(status_code=422, detail="invalid semantic fact payload") from exc

    try:
        result = store(
            text=req.text,
            source=req.source,
            semantic_facts=req.semantic_facts,
        )
    except ValueError as exc:
        # These two graph refusals are invalid typed evidence requests. Do not
        # disguise an internal embedding/state failure as caller input.
        if str(exc) not in (
            "typed write would attach to an existing memory from a different source; "
            "refusing cross-source provenance",
            "typed write collides with a different memory text; "
            "refusing ambiguous provenance",
        ):
            raise
        logger.exception("semantic store refused typed write on provenance conflict")
        raise HTTPException(
            status_code=422,
            detail="typed semantic write refused: provenance conflict",
        ) from exc

    result["fact_ids"] = [fact.id for fact in prepared]
    return result


@app.post("/semantic/recall", dependencies=[Depends(require_semantic_canary)])
async def api_semantic_recall(req: SemanticRecallRequest):
    """Opt-in typed read: every retained version, no winner, no mutation."""
    for field in ("observed_as_of", "valid_at"):
        if field in req.model_fields_set and getattr(req, field) is None:
            raise HTTPException(status_code=422, detail=f"{field} requires an aware ISO timestamp")
    try:
        _semantic._validate_subject_relation(req.subject_id, req.relation_type)
    except ValueError as exc:
        logger.exception("semantic recall rejected subject_id/relation_type")
        raise HTTPException(
            status_code=422, detail="invalid subject_id or relation_type"
        ) from exc

    try:
        observed_as_of = _semantic._utc(req.observed_as_of, "observed_as_of")
        valid_at = _semantic._utc(req.valid_at, "valid_at")
    except ValueError as exc:
        logger.exception("semantic recall rejected observed_as_of/valid_at timestamp")
        raise HTTPException(
            status_code=422, detail="invalid observed_as_of or valid_at timestamp"
        ) from exc

    return recall_facts(
        subject_id=req.subject_id,
        relation_type=req.relation_type,
        observed_as_of=observed_as_of,
        valid_at=valid_at,
    )


def _checked_admin_data_dir() -> str:
    data_dir = os.environ.get("QMG_DATA_DIR", "")
    if not data_dir or os.path.islink(data_dir) or not os.path.isdir(data_dir):
        raise ValueError("DK snapshot requires a regular data directory")
    for name in ("graph.pkl", "graph.json"):
        if os.path.islink(os.path.join(data_dir, name)):
            raise ValueError("DK snapshot rejects symlinked pair files")
    return data_dir


def _inspect_pair_unlocked(data_dir: str):
    """Caller holds _PERSIST_LOCK; independently validate both private files."""
    persisted, doc, _pkl, _json, pair_sha = _rollback._load_pair_checked(
        data_dir, "DK snapshot pair", model=QMG_MODEL)
    live_graph = get_graph()
    _rollback._validate_pair_consistency(live_graph, doc, "DK live vs JSON")
    live = graph_inventory(live_graph)
    disk = graph_inventory(persisted)
    if live != disk:
        raise ValueError("DK live and independently loaded pair inventories differ")
    # A direct-file writer is not serialized by _PERSIST_LOCK. Refuse a swap
    # between loading the disk object and certifying its exact pair bytes.
    if _rollback._pair_digests(_rollback._read_pair_bytes(
            data_dir, "DK post-inventory pair")) != pair_sha:
        raise ValueError("DK pair changed during full-state inventory")
    return live, disk, pair_sha


def _capture_verified_receipt() -> dict:
    """Preflight an existing pair, flush, then prove exact live/disk parity."""
    with _PERSIST_LOCK:
        if _ADMISSION.status()["inflight"]:
            raise RuntimeError("DK HTTP mutations have not drained")
        data_dir = _checked_admin_data_dir()
        pair_paths = [os.path.join(data_dir, name) for name in ("graph.pkl", "graph.json")]
        if any(os.path.lexists(path) for path in pair_paths):
            # Do not overwrite a pre-existing partial/corrupt pair with a
            # plausible new snapshot. A private recovery must handle it first.
            _rollback._load_pair_checked(data_dir, "DK pre-snapshot pair", model=QMG_MODEL)
        # The previous process may have persisted a huge executable model
        # cache. Verify exact named-model weights/tokenizer/config and probes
        # BEFORE detaching; no fact, memory embedding or learned state changes.
        cache_proof = verify_and_detach_embedder(get_graph())
        _persist_current_graph()  # same re-entrant lock; no auto-save interleave
        live, disk, pair_sha = _inspect_pair_unlocked(data_dir)
        if graph_inventory(get_graph()) != live or _ADMISSION.status()["inflight"]:
            raise RuntimeError("DK live state changed during snapshot")
        return {
            "schema": "qmg-semantic-receipt-v2",
            "live": live,
            "persisted": disk,
            "pair_sha256": pair_sha,
            "cache_elision_sha256": cache_proof,
            "denied_writes": _ADMISSION.status()["denied_writes"],
            "inflight": 0,
            "verified": True,
        }


def _recheck_receipt(receipt: dict) -> None:
    """Never trust a stale receipt when resuming or reading verification."""
    with _PERSIST_LOCK:
        data_dir = _checked_admin_data_dir()
        live, disk, pair_sha = _inspect_pair_unlocked(data_dir)
        if (receipt["live"] != live or receipt["persisted"] != disk
                or receipt["pair_sha256"] != pair_sha
                or _ADMISSION.status()["inflight"]):
            raise ValueError("DK receipt no longer matches live and persisted state")


async def _admin_json(request: Request) -> dict:
    """Validate controls without Pydantic echoing a secret in a 422 body."""
    try:
        raw = await request.body()
        body = json.loads(raw) if raw else {}
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(status_code=422, detail="Malformed admin JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Malformed admin JSON")
    return body


@app.get("/semantic/admin/status", dependencies=[Depends(require_admin_canary)])
async def api_semantic_admin_status():
    return _ADMISSION.status()


@app.post("/semantic/admin/quiesce", dependencies=[Depends(require_admin_canary)])
async def api_semantic_admin_quiesce(request: Request):
    body = await _admin_json(request)
    timeout = body.get("timeout", 30.0)
    if (set(body) - {"timeout"} or type(timeout) not in (int, float)
            or not math.isfinite(timeout) or not 0.01 <= timeout <= 30.0):
        raise HTTPException(status_code=422, detail="Invalid bounded drain timeout")
    try:
        _ADMISSION.close()  # atomic with every HTTP admission decision
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail="Maintenance already active") from exc
    receipt = None
    try:
        if not await asyncio.to_thread(_ADMISSION.wait_drained, float(timeout)):
            raise HTTPException(status_code=503,
                                detail="Ingress drain timed out; gate remains closed")
        try:
            receipt = await asyncio.to_thread(_capture_verified_receipt)
        except Exception as exc:
            raise HTTPException(status_code=409,
                                detail="Snapshot verification failed; gate remains closed") from exc
    finally:
        # Any timeout, corruption, failure, or cancellation leaves ingress
        # CLOSED with no verified receipt. No implicit retry or reopen.
        _ADMISSION.finish_pause(receipt)
    return {"ok": True, "paused": True, "receipt": _ADMISSION.current_receipt()}


@app.get("/semantic/admin/receipt", dependencies=[Depends(require_admin_canary)])
async def api_semantic_admin_receipt():
    current = _ADMISSION.current_receipt()
    if current is None:
        return {"ok": False, "paused": _ADMISSION.is_paused(), "receipt": None}
    try:
        await asyncio.to_thread(_recheck_receipt, current)
    except Exception:
        _ADMISSION.invalidate_receipt()
        return {"ok": False, "paused": True, "receipt": None}
    latest = _ADMISSION.current_receipt()
    if latest is None or latest["pair_sha256"] != current["pair_sha256"]:
        return {"ok": False, "paused": _ADMISSION.is_paused(), "receipt": None}
    return {"ok": True, "paused": True, "receipt": latest}


@app.post("/semantic/admin/resume", dependencies=[Depends(require_admin_canary)])
async def api_semantic_admin_resume(request: Request):
    body = await _admin_json(request)
    receipt_sha = body.get("receipt_sha256")
    if (set(body) != {"receipt_sha256"} or not isinstance(receipt_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", receipt_sha) is None):
        raise HTTPException(status_code=422, detail="Invalid receipt SHA-256")
    try:
        receipt = _ADMISSION.start_resume(receipt_sha)
    except RuntimeError as exc:
        raise HTTPException(status_code=409,
                            detail="Resume refused; verify receipt and denied-write audit") from exc
    verified = False
    try:
        await asyncio.to_thread(_recheck_receipt, receipt)
        verified = True
    except Exception as exc:
        raise HTTPException(status_code=409,
                            detail="Resume verification failed; gate remains closed") from exc
    finally:
        reopened = _ADMISSION.finish_resume(verified, receipt_sha)
    if not reopened:
        raise HTTPException(status_code=409,
                            detail="Resume refused; denied-write audit is NO-GO")
    return {"ok": True, "paused": False, "denied_writes": 0}


@app.post("/reset")
async def api_reset():
    """Replace all in-memory state and delete persisted graph snapshots."""
    global _graph
    with _PERSIST_LOCK:
        current = get_graph()
        _graph = MemoryGraph(
            similarity_threshold=current.similarity_threshold,
            model=current._model_name,
        )
        set_graph(_graph)
        set_stm(ShortTermMemory())
        data_dir = os.environ.get("QMG_DATA_DIR", "")
        if data_dir:
            for filename in ("graph.pkl", "graph.json"):
                try:
                    os.remove(os.path.join(data_dir, filename))
                except FileNotFoundError:
                    pass
    return {"ok": True, "nodes": 0}


class StoreBatchRequest(BaseModel):
    texts: List[str]
    sources: Optional[List[str]] = None


class AnswerRequest(BaseModel):
    query: str
    K: int = 5


class TemporalAnswerRequest(BaseModel):
    query: str
    K: int = 8


@app.post("/store-batch")
async def api_store_batch(req: StoreBatchRequest):
    result = store_batch(
        texts=req.texts,
        sources=req.sources,
    )
    return result


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    text_weight: float = 0.3


@app.post("/search")
async def api_search(req: SearchRequest):
    g = get_graph()
    results = hybrid_search(
        query=req.query,
        graph=g,
        k=req.k,
        text_weight=req.text_weight,
        embed_weight=1.0 - req.text_weight,
    )
    return {"ok": True, "query": req.query, "results": results, "total": len(results)}


class RecallRequest(BaseModel):
    query: str
    k: int = 5
    hops: int = 2
    top_seeds: int = 5
    alpha: float = 0.4
    beta_conn: float = 0.35
    gamma_cov: float = 0.25
    max_candidates: int = 14
    text_weight: float = 0.3
    use_recency: bool = True


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
        text_weight=req.text_weight,
        use_recency=req.use_recency,
    )
    # Recall tracker — auto-counts queries, triggers IBM batch at 25
    _track_recall(req.query)
    # Log recall pattern for self-learning
    if _graph is not None and result.get("memories"):
        try:
            node_ids = [m.get("id", "") for m in result["memories"] if m.get("id")]
            _graph.recall_tracker.log_recall(req.query, node_ids)
        except Exception:
            pass
    return result


@app.post("/learn")
async def api_learn():
    """Process recall history and update learned edge weights."""
    if _graph is None:
        return {"ok": False, "error": "Graph not loaded"}
    try:
        weights = _graph.learn(min_co_recalls=1)
        mem_scores = _graph.get_memory_scores()
        return {
            "ok": True,
            "learned_edges": len(weights),
            "scored_memories": len(mem_scores),
            "top_edges": {f"{a[:8]}↔{b[:8]}": round(w, 3) for (a,b), w in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10] if w > 0},
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/scores")
async def api_memory_scores():
    """Get memory value scores based on recall frequency."""
    if _graph is None:
        return {"ok": False, "error": "Graph not loaded"}
    try:
        scores = _graph.get_memory_scores()
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            "ok": True,
            "scored_memories": len(scores),
            "top": [{"id": mid[:8], "score": round(s, 3)} for mid, s in top],
            "zero_score_count": sum(1 for v in scores.values() if v == 0),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


_TRACKER_DATA_DIR = None

def _get_tracker_dir():
    global _TRACKER_DATA_DIR
    if _TRACKER_DATA_DIR is None:
        d = os.environ.get("QMG_DATA_DIR", "/tmp/.qmg-tracker")
        # Fall back to /tmp if data dir isn't writable
        if not os.access(d, os.W_OK):
            d = os.path.join("/tmp/.qmg-tracker", os.path.basename(d))
        _TRACKER_DATA_DIR = d
    return _TRACKER_DATA_DIR

def _track_recall(query: str):
    """Increment recall counter, dump batch at 25."""
    try:
        d = _get_tracker_dir()
        os.makedirs(d, exist_ok=True)
        tp = os.path.join(d, "recall_tracker.json")
        bp = os.path.join(d, "ibm_batch_queries.json")
        
        tracker = {"count": 0, "queries": []}
        if os.path.exists(tp):
            with open(tp) as f:
                tracker = json.load(f)
        
        tracker["count"] += 1
        tracker["queries"].append({
            "query": query,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        })
        
        with open(tp, "w") as f:
            json.dump(tracker, f, indent=2)
        
        if tracker["count"] >= 25:
            with open(bp, "w") as f:
                json.dump(tracker["queries"], f, indent=2)
            print(f"  [recall-tracker] {tracker['count']} queries — batch saved to {bp}")
            # Reset
            tracker = {"count": 0, "queries": [], "last_batch_time": __import__("datetime").datetime.utcnow().isoformat()}
            with open(tp, "w") as f:
                json.dump(tracker, f, indent=2)
    except Exception as e:
        print(f"  [recall-tracker] WARNING: {e}")


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


class MergeRequest(BaseModel):
    memory_ids: List[str]


@app.post("/merge")
async def api_merge(req: MergeRequest):
    """Merge duplicate memories into one survivor node."""
    g = get_graph()
    if not req.memory_ids or len(req.memory_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 memory IDs to merge")
    result = g.merge_memories(req.memory_ids)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Merge failed"))
    return result


class FindDuplicatesRequest(BaseModel):
    threshold: float = 0.95


class MergeDuplicatesRequest(BaseModel):
    threshold: float = 0.92
    dry_run: bool = False
    max_clusters: Optional[int] = None


@app.post("/merge-duplicates")
async def api_merge_duplicates(req: MergeDuplicatesRequest):
    """Find duplicate clusters (cosine >= threshold) and merge each into a
    single survivor. Wraps find_duplicates + merge_clusters. dry_run reports
    what would merge without changing the graph."""
    g = get_graph()
    if not g.memories:
        return {"ok": True, "clusters_merged": 0, "memories_merged": 0, "dry_run": req.dry_run}
    # Reuse the same clustering logic as /find-duplicates
    import numpy as np
    ids = list(g.memories.keys())
    embs = np.array([g.memories[i].embedding for i in ids])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs_n = embs / norms
    sim = embs_n @ embs_n.T
    n = len(ids)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    thr = req.threshold
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= thr:
                union(i, j)
    clusters_map = {}
    for i in range(n):
        clusters_map.setdefault(find(i), []).append(ids[i])
    clusters = [c for c in clusters_map.values() if len(c) > 1]
    if req.max_clusters:
        clusters = clusters[:req.max_clusters]

    result = g.merge_clusters(clusters, dry_run=req.dry_run)
    result["threshold"] = thr
    result["n_clusters_found"] = len(clusters)
    if not req.dry_run:
        # Persist after a real merge so the cleanup survives restart
        try:
            _persist_current_graph()
            result["persisted"] = True
        except Exception as e:
            result["persisted"] = False
            result["persist_error"] = str(e)
    return result


@app.post("/find-duplicates")
async def api_find_duplicates(req: FindDuplicatesRequest):
    """Find duplicate memories by cosine similarity."""
    g = get_graph()
    if not g.memories:
        return {"ok": True, "clusters": []}

    mem_ids = list(g.memories.keys())
    mems = [g.memories[mid] for mid in mem_ids]

    # Collect valid embeddings
    valid = [(mid, mem) for mid, mem in zip(mem_ids, mems)
             if mem.embedding is not None]
    
    if len(valid) < 2:
        return {"ok": True, "clusters": []}
    
    ids, embeddings = zip(*[(v[0], v[1].embedding) for v in valid])
    mat = np.stack(embeddings)
    sim = np.dot(mat, mat.T)
    
    # Find clusters (similarity > threshold)
    clustered = set()
    clusters = []
    for i in range(len(ids)):
        if i in clustered:
            continue
        group = [j for j in range(i + 1, len(ids)) 
                 if sim[i][j] > req.threshold and j not in clustered]
        if group:
            cluster = [ids[i]] + [ids[j] for j in group]
            clusters.append(cluster)
            clustered.update([i] + group)
    
    return {
        "ok": True,
        "clusters": clusters,
        "n_clusters": len(clusters),
        "total_duplicates": sum(len(c) - 1 for c in clusters),
    }


class DeleteBySourceRequest(BaseModel):
    sources: List[str]
    dry_run: bool = False


@app.post("/delete-by-source")
async def api_delete_by_source(req: DeleteBySourceRequest):
    """Delete memories whose source matches (exact or prefix) any in
    `sources`. dry_run reports what would be removed without changing the
    graph. Persists after a real delete so the cleanup survives restart."""
    if not req.sources:
        raise HTTPException(status_code=400, detail="Need at least one source")
    g = get_graph()
    if not g.memories:
        return {"ok": True, "matched": 0, "removed": 0, "dry_run": req.dry_run}
    result = g.delete_by_source(req.sources, dry_run=req.dry_run)
    if not req.dry_run:
        try:
            _persist_current_graph()
            result["persisted"] = True
        except Exception as e:
            result["persisted"] = False
            result["persist_error"] = str(e)
    return result


@app.get("/graph")
async def api_full_graph(min_weight: float = 0.6, max_edges: int = 2000):
    g = get_graph()
    nodes = []
    edges = []

    # Add memory nodes
    for mid, mem in g.memories.items():
        nodes.append({
            "id": mid,
            "type": "memory",
            "text": mem.text[:100],
            "source": mem.source or "unknown",
            "entities": mem.entities[:5],
            "ts": mem.timestamp.isoformat() if mem.timestamp else None,
            "degree": g.graph.degree(mid) if mid in g.graph else 0,
        })

    # Source nodes exist only on newer graphs. Older pickles predate the
    # source-node registry, so graph visualization must remain compatible.
    sources = getattr(g, "sources", {})
    for sid, src in sources.items():
        mem_count = sum(1 for n in g.graph.neighbors(sid)
                       if g.graph.nodes[n].get("node_type") != "source")
        nodes.append({
            "id": sid,
            "type": "source",
            "source_type": src.type,
            "url": src.url,
            "title": src.title,
            "date_added": src.date_added,
            "memories": mem_count,
            "degree": g.graph.degree(sid) if sid in g.graph else 0,
        })

    # Add edges — sorted by weight, filtered, capped
    all_edges = []
    for u, v, data in g.graph.edges(data=True):
        w = data.get("weight", 0)
        if w >= min_weight:
            all_edges.append({
                "source": u,
                "target": v,
                "weight": round(w, 3),
                "types": data.get("types", []),
            })
    all_edges.sort(key=lambda e: e["weight"], reverse=True)
    edges = all_edges[:max_edges]

    return {
        "ok": True,
        "nodes": nodes,
        "edges": edges,
        "stats": g.stats(),
        "source_count": len(sources),
        "filters": {"min_weight": min_weight, "max_edges": max_edges, "total_edges": len(all_edges)},
    }


@app.get("/graph-view")
async def graph_viewer():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QMG Source Graph</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0d1117; font-family: system-ui, -apple-system, sans-serif; overflow: hidden; color: #c9d1d9; }
#graph { width: 100vw; height: 100vh; }
.sidebar {
  position: fixed; right: 0; top: 0; width: 340px; height: 100vh;
  background: #161b22; border-left: 1px solid #30363d;
  padding: 20px; overflow-y: auto; z-index: 10;
  transition: transform 0.3s ease;
}
.sidebar.hidden { transform: translateX(100%); }
.sidebar h3 { color: #58a6ff; margin-bottom: 8px; font-size: 15px; }
.sidebar .meta { font-size: 12px; color: #8b949e; margin-bottom: 12px; }
.sidebar .text { font-size: 13px; line-height: 1.6; color: #c9d1d9; }
.sidebar .entities { margin-top: 12px; }
.sidebar .entities span {
  display: inline-block; background: #21262d; color: #7ee787;
  padding: 2px 8px; border-radius: 12px; margin: 2px 4px 2px 0; font-size: 11px;
}
.sidebar .close-btn {
  position: absolute; top: 12px; right: 16px;
  background: none; border: none; color: #8b949e; font-size: 20px; cursor: pointer;
}
.sidebar .close-btn:hover { color: #f85149; }
.legend {
  position: fixed; bottom: 16px; left: 16px;
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 12px 16px; font-size: 12px; z-index: 10;
}
.legend h4 { color: #8b949e; margin-bottom: 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
.legend-item { display: flex; align-items: center; margin: 4px 0; }
.legend-swatch { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }
.legend-swatch.diamond { border-radius: 2px; transform: rotate(45deg); width: 10px; height: 10px; margin: 0 9px 0 1px; }
.legend-label { color: #c9d1d9; }
.tooltip {
  position: fixed; pointer-events: none; background: #21262d; color: #c9d1d9;
  padding: 6px 12px; border-radius: 6px; font-size: 12px; border: 1px solid #30363d;
  opacity: 0; transition: opacity 0.15s; z-index: 20; max-width: 280px;
}
.stats-bar {
  position: fixed; top: 12px; left: 16px;
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 8px 14px; font-size: 12px; z-index: 10; color: #8b949e;
}
.stats-bar strong { color: #58a6ff; }
.slider-bar {
  position: fixed; top: 12px; right: 360px;
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 8px 14px; font-size: 12px; z-index: 10; color: #8b949e;
  display: flex; align-items: center; gap: 8px;
}
.slider-bar input[type=range] { width: 100px; accent-color: #58a6ff; }
.slider-bar .val { color: #58a6ff; font-weight: bold; min-width: 28px; }
.loading {
  position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
  color: #58a6ff; font-size: 18px; z-index: 5;
}
</style>
</head>
<body>
<div id="graph"></div>
<div class="loading" id="loading">Loading graph data...</div>
<div class="sidebar hidden" id="sidebar">
  <button class="close-btn" onclick="closeSidebar()">&times;</button>
  <h3 id="side-title"></h3>
  <div class="meta" id="side-meta"></div>
  <div class="text" id="side-text"></div>
  <div class="entities" id="side-entities"></div>
</div>
<div class="tooltip" id="tooltip"></div>
<div class="stats-bar" id="stats"></div>
<div class="slider-bar">
  <span>Weight ≥</span>
  <input type="range" id="weightSlider" min="0.3" max="1.0" step="0.05" value="0.6">
  <span class="val" id="weightVal">0.6</span>
</div>
<div class="legend" id="legend"><h4>Legend</h4></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const COLORS = [
  "#58a6ff","#7ee787","#f0883e","#f778ba","#d2a8ff",
  "#79c0ff","#a5d6ff","#ffa657","#ff7b72","#56d364",
  "#e3b341","#d876e3","#768390","#c96198","#e55c31"
];
const srcColor = d3.scaleOrdinal(COLORS);
const sidebar = document.getElementById("sidebar");
const tooltip = document.getElementById("tooltip");
let currentSvg = null;

function closeSidebar() { sidebar.classList.add("hidden"); }

function loadGraph(minWeight) {
  // Clear previous
  if (currentSvg) { currentSvg.remove(); currentSvg = null; }
  document.getElementById("loading").style.display = "block";
  document.getElementById("legend").innerHTML = "<h4>Legend</h4>";
  
  fetch("/graph?min_weight=" + minWeight + "&max_edges=2000")
    .then(r => r.json())
    .then(data => {
      document.getElementById("loading").style.display = "none";
      if (!data.ok || !data.nodes.length) {
        document.getElementById("loading").textContent = "No graph data at weight ≥ " + minWeight;
        document.getElementById("loading").style.display = "block";
        return;
      }
      const f = data.filters || {};
      document.getElementById("stats").innerHTML =
        "Nodes: <strong>" + data.nodes.length + "</strong> &nbsp;|&nbsp; " +
        "Edges: <strong>" + data.edges.length + "</strong> &nbsp;|&nbsp; " +
        "Sources: <strong>" + (data.source_count || 0) + "</strong>" +
        (f.total_edges > data.edges.length ? " (of " + f.total_edges + ")" : "");

      // Build legend
      const sources = new Map();
      data.nodes.filter(n => n.type === "source").forEach(n =>
        sources.set(n.title || n.id, { color: srcColor(n.title || n.id), type: n.source_type }));
      const leg = document.getElementById("legend");
      sources.forEach((v, k) => {
        const div = document.createElement("div");
        div.className = "legend-item";
        div.innerHTML = '<div class="legend-swatch diamond" style="background:' + v.color + '"></div><span class="legend-label">' + k + '</span>';
        leg.appendChild(div);
      });
      const memDiv = document.createElement("div");
      memDiv.className = "legend-item";
      memDiv.innerHTML = '<div class="legend-swatch" style="background:#8b949e"></div><span class="legend-label">Memory node</span>';
      leg.appendChild(memDiv);

      renderGraph(data);
    })
    .catch(e => {
      document.getElementById("loading").textContent = "Error: " + e.message;
    });
}

// Weight slider
const slider = document.getElementById("weightSlider");
const weightVal = document.getElementById("weightVal");
slider.addEventListener("input", () => {
  weightVal.textContent = parseFloat(slider.value).toFixed(2);
});
slider.addEventListener("change", () => {
  loadGraph(parseFloat(slider.value));
});

// Initial load
loadGraph(0.6);

function renderGraph(data) {
  const W = window.innerWidth, H = window.innerHeight;
  const svg = d3.select("#graph").append("svg")
    .attr("width", W).attr("height", H);

  const g = svg.append("g");

  // Zoom/pan
  svg.call(d3.zoom().scaleExtent([0.1, 4]).on("zoom", (event) => {
    g.attr("transform", event.transform);
  }));

  // Build lookup
  const nodeMap = new Map(data.nodes.map(n => [n.id, n]));

  // Node size scale
  const maxDeg = d3.max(data.nodes, d => d.degree || 0) || 1;
  const nodeRadius = d => d.type === "source" ? 12 : 3 + (d.degree / maxDeg) * 18;

  // Color function
  function getColor(d) {
    if (d.type === "source") return srcColor(d.title || d.id);
    return srcColor(d.source || "unknown");
  }

  // Shape function
  function drawNode(sel) {
    sel.each(function(d) {
      const el = d3.select(this);
      if (d.type === "source") {
        const r = nodeRadius(d);
        el.attr("d", d3.symbol().type(d3.symbolDiamond).size(r * r * 4)());
      } else {
        el.attr("r", nodeRadius(d));
      }
    });
  }

  // Simulation
  const sim = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.edges).id(d => d.id).distance(80).strength(0.3))
    .force("charge", d3.forceManyBody().strength(-120))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collision", d3.forceCollide().radius(d => nodeRadius(d) + 4));

  // Links
  const link = g.append("g").selectAll("line")
    .data(data.edges).join("line")
    .attr("stroke", "#30363d")
    .attr("stroke-width", d => Math.max(0.3, d.weight * 3))
    .attr("stroke-opacity", d => Math.max(0.1, d.weight));

  // Nodes - memory as circle, source as path (diamond)
  const nodeG = g.append("g").selectAll("g")
    .data(data.nodes).join("g")
    .call(d3.drag()
      .on("start", (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end", (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

  // Memory nodes: circles
  nodeG.filter(d => d.type === "memory").append("circle")
    .attr("fill", d => getColor(d))
    .attr("stroke", "#0d1117")
    .attr("stroke-width", 1.5)
    .attr("r", d => nodeRadius(d))
    .attr("cursor", "pointer");

  // Source nodes: diamond path
  nodeG.filter(d => d.type === "source").append("path")
    .attr("fill", d => getColor(d))
    .attr("stroke", "#f0f6fc")
    .attr("stroke-width", 2)
    .attr("d", d3.symbol().type(d3.symbolDiamond).size(12 * 12 * 4)())
    .attr("cursor", "pointer");

  // Interaction
  nodeG.on("click", (event, d) => {
    event.stopPropagation();
    showSidebar(d);
  }).on("mouseenter", (event, d) => {
    const label = d.type === "source" ? (d.title || d.url || d.id) : (d.text || d.id);
    tooltip.style("opacity", 1).html(
      "<strong>" + d.type + "</strong>: " + label.slice(0, 120) +
      (d.type === "source" ? "<br>Memories: " + (d.memories || 0) : "")
    );
  }).on("mousemove", (event) => {
    tooltip.style("left", (event.pageX + 12) + "px").style("top", (event.pageY - 10) + "px");
  }).on("mouseleave", () => tooltip.style("opacity", 0));

  // Click background to close sidebar
  svg.on("click", () => closeSidebar());

  sim.on("tick", () => {
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    nodeG.attr("transform", d => "translate(" + d.x + "," + d.y + ")");
  });

  function showSidebar(d) {
    document.getElementById("side-title").textContent = d.type === "source"
      ? (d.title || d.url || d.id) : (d.text ? d.text.slice(0, 60) + "..." : d.id);
    const meta = [];
    meta.push("Type: <strong>" + d.type + "</strong>");
    if (d.type === "source") {
      if (d.source_type) meta.push("Source type: " + d.source_type);
      if (d.url) meta.push("URL: " + d.url);
      if (d.date_added) meta.push("Added: " + d.date_added);
      meta.push("Connected memories: " + (d.memories || 0));
    } else {
      meta.push("Source: " + (d.source || "unknown"));
      if (d.ts) meta.push("Timestamp: " + d.ts);
      meta.push("Connections: " + (d.degree || 0));
    }
    document.getElementById("side-meta").innerHTML = meta.join(" &nbsp;|&nbsp; ");
    document.getElementById("side-text").textContent = d.text || "";
    const entDiv = document.getElementById("side-entities");
    entDiv.innerHTML = "";
    if (d.entities && d.entities.length) {
      d.entities.forEach(e => {
        const span = document.createElement("span");
        span.textContent = e;
        entDiv.appendChild(span);
      });
    }
    sidebar.classList.remove("hidden");
  }
}
</script>
</body>
</html>"""
    return Response(content=html, media_type="text/html")


@app.post("/answer")
async def api_answer(req: AnswerRequest):
    result = answer(
        query=req.query,
        K=req.K,
    )
    return result


@app.post("/temporal-answer")
async def api_temporal_answer(req: TemporalAnswerRequest):
    result = temporal_answer(
        query=req.query,
        K=req.K,
    )
    return result


@app.post("/prune")
async def api_prune():
    g = get_graph()
    result = g.prune_existing()
    return result


def main():
    host = os.environ.get("QMG_HOST", "0.0.0.0")
    port = int(os.environ.get("QMG_PORT", "8502"))
    print(f"⚛️🧠 Quantum Memory Graph API starting on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
