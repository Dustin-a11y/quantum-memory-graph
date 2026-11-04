#!/usr/bin/env python3
"""Kong Graph MCP server — stdio JSON-RPC, no hardcoded endpoints or secrets.

Every endpoint and token comes from the environment, so the same wheel serves any agent:

    QMG_AGENT                 name used to tag this agent's memories   (default: "agent")
    QMG_URL / QMG_TOKEN       personal graph                          (default: http://127.0.0.1:8503)
    QMG_HISTORICAL_URL/_TOKEN historical archive fallback             (default: http://127.0.0.1:8502)
    QMG_BENCHD_URL/_TOKEN     dev/bench dual-write                    (default: http://127.0.0.1:8518)
    QMG_MEGABRAIN_URL/_TOKEN  fleet combined graph                    (default: http://127.0.0.1:8520)

Unset optional URLs/tokens simply disable that hop — nothing raises at import time. This is the
packaged version of the profile script that used to carry four plaintext API tokens.
"""
import json
import sys
import requests
import os
import subprocess
from urllib3.util.retry import Retry

def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


AGENT = _env("QMG_AGENT", "agent")
PERSONAL = _env("QMG_URL", "http://127.0.0.1:8503")
QMG_TOKEN = _env("QMG_TOKEN")
HISTORICAL = _env("QMG_HISTORICAL_URL", "http://127.0.0.1:8502")
HISTORICAL_TOKEN = _env("QMG_HISTORICAL_TOKEN")
BENCHD = _env("QMG_BENCHD_URL", "http://127.0.0.1:8518")
BENCHD_TOKEN = _env("QMG_BENCHD_TOKEN")
MEGABRAIN = _env("QMG_MEGABRAIN_URL", "http://127.0.0.1:8520")
MEGABRAIN_TOKEN = _env("QMG_MEGABRAIN_TOKEN")


def _enabled(url: str, token: str) -> bool:
    return bool(url and token)

_session = None

def get_session():
    global _session
    if _session is None:
        from requests.adapters import HTTPAdapter
        _session = requests.Session()
        retries = Retry(total=1, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        _session.mount("http://", HTTPAdapter(max_retries=retries))
    return _session

def _call(method: str, url: str, path: str, token: str = QMG_TOKEN, **kwargs) -> dict:
    s = get_session()
    headers = kwargs.pop("headers", {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = s.request(method, f"{url}{path}", timeout=15, headers=headers, **kwargs)
        if r.status_code == 200:
            return r.json()
    except (requests.ConnectionError, requests.Timeout):
        pass
    return None

def store(text: str, source: str = AGENT) -> dict:
    """Store to personal graph + dual-write to benchd + mega brain."""
    result = _call("POST", PERSONAL, "/store", json={"text": text, "source": source})
    # Fire-and-forget dual-write to benchd dev instance
    try:
        if not _enabled(BENCHD, BENCHD_TOKEN):
            raise RuntimeError("benchd not configured")
        _call("POST", BENCHD, "/store", token=BENCHD_TOKEN, json={"text": text, "source": source})
    except:
        pass  # Never block on benchd write
    # Fire-and-forget to Mega Brain with agent tag
    try:
        if not _enabled(MEGABRAIN, MEGABRAIN_TOKEN):
            raise RuntimeError("mega brain not configured")
        _call("POST", MEGABRAIN, "/store", token=MEGABRAIN_TOKEN, json={"text": f"[{AGENT}] {text}", "source": f"{AGENT}-{source}"})
    except:
        pass  # Never block on mega brain write
    if result:
        return result
    return {"error": "Kong Graph unavailable"}

def recall(query: str, k: int = 5) -> dict:
    """Semantic recall with an honest relevance score.

    /search blends lexical and embedding similarity and orders by it; /recall selects a QAOA
    subgraph by graph structure and reports relevance near 1.0 for unrelated fragments, which an
    agent reads as confident memory. Search is therefore primary and /recall is the fallback for
    when search is empty (a structurally-adjacent memory still beats nothing). Every returned
    memory carries `relevance` = the similarity actually measured, so a weak hit looks weak.
    """
    s = get_session()

    memories = []
    try:
        searched = _call("POST", PERSONAL, "/search",
                         token=QMG_TOKEN, json={"query": query, "k": k})
        for item in (searched or {}).get("results") or []:
            text = item.get("text") or ""
            if not text:
                continue
            embed = float(item.get("embed_score") or 0.0)
            lex = float(item.get("text_score") or 0.0)
            # Blend, never max(): text_score saturates at 1.0 on a single shared word
            # ("Dustin asked what to do." scored 1.0 for "...what did Dustin decide about
            # memory?"), so a max() let lexical noise outrank semantic match.
            blended = round(0.8 * embed + 0.2 * lex, 4)
            memories.append({
                "id": item.get("id"),
                "text": text,
                "entities": item.get("entities") or [],
                "relevance": blended,
                "source": item.get("source"),
                "tier": item.get("tier"),
                "timestamp": item.get("timestamp"),
                "metadata": item.get("metadata") or {},
                "connections": item.get("connections") or [],
            })
    except Exception:
        memories = []

    if memories:
        # Rank here: the API's ordering is a candidate order, not a similarity order.
        memories.sort(key=lambda m: m["relevance"], reverse=True)
        return {"ok": True, "memories": memories[:k], "method": "search", "cascade": "personal"}

    personal_result = _call("POST", PERSONAL, "/recall",
                            token=QMG_TOKEN, json={"query": query, "k": k})
    good_results = False
    if personal_result and personal_result.get("memories"):
        for m in personal_result["memories"]:
            if (m.get("relevance") or 0) > 0.4:
                good_results = True
                break

    if good_results:
        result = personal_result
        result["cascade"] = "personal"
    else:
        historical_result = _call("POST", HISTORICAL, "/recall",
                                  token=HISTORICAL_TOKEN, json={"query": query, "k": k})
        if historical_result and historical_result.get("memories"):
            result = historical_result
            result["cascade"] = "historical"
        else:
            result = personal_result or {"ok": False, "memories": []}
            result["cascade"] = "none"

    return result


def stats() -> dict:
    s = get_session()
    try:
        r = s.get(f"{PERSONAL}/stats", timeout=10, headers={"Authorization": f"Bearer {QMG_TOKEN}"})
        if r.status_code == 200:
            return r.json()
    except:
        pass
    try:
        r = s.get(f"{HISTORICAL}/stats", timeout=10, headers={"Authorization": f"Bearer {HISTORICAL_TOKEN}"})
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {"nodes": 0, "edges": 0}

# ─── Source API functions ────────────────────────────────────────

def add_source(source_type: str, url: str = "", title: str = "", content: str = "") -> dict:
    """Create a source node in Kong Graph."""
    s = get_session()
    try:
        r = s.post(f"{PERSONAL}/source", json={
            "type": source_type, "url": url, "title": title, "content": content
        }, timeout=10, headers={"Authorization": f"Bearer {QMG_TOKEN}"})
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {"error": "Kong Graph unavailable"}

def list_sources() -> dict:
    """List all source nodes."""
    s = get_session()
    try:
        r = s.get(f"{PERSONAL}/sources", timeout=10, headers={"Authorization": f"Bearer {QMG_TOKEN}"})
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {"ok": False, "sources": []}

def source_graph(source_id: str) -> dict:
    """Get memories connected to a source node."""
    s = get_session()
    try:
        r = s.get(f"{PERSONAL}/source/{source_id}/graph", timeout=10, headers={"Authorization": f"Bearer {QMG_TOKEN}"})
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {"error": "Kong Graph unavailable"}

def full_graph() -> dict:
    """Get full graph as D3-compatible JSON."""
    s = get_session()
    try:
        r = s.get(f"{PERSONAL}/graph", timeout=15, headers={"Authorization": f"Bearer {QMG_TOKEN}"})
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {"ok": False, "nodes": [], "edges": []}

def make_result(data: dict) -> dict:
    """Wrap raw response in proper MCP CallToolResult format."""
    return {"content": [{"type": "text", "text": json.dumps(data)}]}

TOOLS = [
    {
        "name": "qmg_store",
        "description": "Store a memory in Kong Graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The memory text to store"},
                "source": {"type": "string", "description": "Optional source identifier"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "qmg_recall",
        "description": "Recall memories from Kong Graph (classical relevance search; QAOA is opt-in via QMG_RECALL_METHOD)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "k": {"type": "integer", "description": "Number of results (default 5)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "qmg_stats",
        "description": "Get Kong Graph statistics (nodes, edges, memory count)",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "qmg_add_source",
        "description": "Add a source node to Kong Graph (URL, file, or manual reference)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "Source type: url, file, or manual"},
                "url": {"type": "string", "description": "URL of the source (optional)"},
                "title": {"type": "string", "description": "Title of the source (optional)"},
                "content": {"type": "string", "description": "Content hash or description (optional)"}
            },
            "required": ["type"]
        }
    },
    {
        "name": "qmg_list_sources",
        "description": "List all source nodes in Kong Graph with their memory counts",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "qmg_source_graph",
        "description": "Get all memories connected to a specific source node in Kong Graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "The source node ID"}
            },
            "required": ["source_id"]
        }
    },
    {
        "name": "qmg_graph",
        "description": "Get full Kong Graph graph as D3-compatible JSON for visualization (nodes, edges, stats)",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]

def handle_request(req: dict) -> dict:
    """Handle MCP JSON-RPC request. Returns dict with 'result' or 'error' + 'id'."""
    req_id = req.get("id", None)
    method = req.get("method", "")

    # MCP initialization handshake
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "qmg-mcp", "version": "0.1.0"},
                "capabilities": {
                    "tools": {},
                    "logging": {},
                    "experimental": {}
                }
            }
        }

    # Notification — no response expected
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None

    # Tool requests
    if method in ("tools/call",) or method in ("qmg_store", "qmg_recall", "qmg_stats"):
        tool_name = req.get("params", {}).get("name", method)
        params = req.get("params", {}).get("arguments", req.get("params", {}))

        if tool_name == "qmg_store":
            result = store(params.get("text",""), params.get("source", AGENT))
        elif tool_name == "qmg_recall":
            result = recall(params.get("query",""), params.get("k", 5))
        elif tool_name == "qmg_stats":
            result = stats()
        elif tool_name == "qmg_add_source":
            result = add_source(params.get("type",""), params.get("url",""), params.get("title",""), params.get("content",""))
        elif tool_name == "qmg_list_sources":
            result = list_sources()
        elif tool_name == "qmg_source_graph":
            result = source_graph(params.get("source_id",""))
        elif tool_name == "qmg_graph":
            result = full_graph()
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {tool_name}"}
            }

        return {"jsonrpc": "2.0", "id": req_id, "result": make_result(result)}

    # tools/list
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS}
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"}
    }

def main() -> int:
    """Run the MCP server over stdio: one JSON-RPC request per line, one response per line."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            if resp is not None:
                print(json.dumps(resp), flush=True)
        except Exception as e:
            print(json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}, "id": None}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
