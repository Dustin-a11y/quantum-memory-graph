"""
Ollama + QMG Memory Middleware
===============================
Transparent proxy that adds graph-based memory to any Ollama model.
Send requests to localhost:11435 instead of :11434 — the middleware
injects recalled QMG memories into the system prompt and stores every
exchange for future recall.

Requirements:
    pip install quantum-memory-graph flask requests

Usage:
    # Terminal 1 — Start Ollama (if not already running)
    ollama serve

    # Terminal 2 — Start the middleware
    python ollama_qmg_middleware.py

    # Terminal 3 — Use it (point any Ollama client at :11435)
    curl http://localhost:11435/api/chat -d '{
      "model": "llama3.2:3b",
      "messages": [{"role": "user", "content": "What do you remember about me?"}]
    }'

    # Or with the Ollama CLI (via OLLAMA_HOST)
    OLLAMA_HOST=http://localhost:11435 ollama run llama3.2:3b

Architecture:
    ┌──────────┐      ┌─────────────────┐      ┌──────────┐
    │  Client   │─────▶│ QMG Middleware  │─────▶│  Ollama   │
    │ (port     │      │ (port 11435)    │      │ (11434)   │
    │  client)  │◀─────│                 │◀─────│          │
    └──────────┘      └───────┬─────────┘      └──────────┘
                              │
                     ┌────────▼────────┐
                     │   QMG Graph      │
                     │  store / recall  │
                     └─────────────────┘

Flow per request:
  1. Extract last user message
  2. recall() from QMG — get K most relevant past memories
  3. Inject memories as a system message at the front of the message list
  4. Forward augmented request to Ollama
  5. Collect response (handles both streaming and non-streaming)
  6. store() the user message and assistant reply into QMG
  7. Return response to client
"""

import json
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# QMG import — graceful fallback
# ---------------------------------------------------------------------------
try:
    from quantum_memory_graph import store, recall
    QMG_AVAILABLE = True
except ImportError:
    QMG_AVAILABLE = False
    print("[WARNING] quantum-memory-graph not installed.")
    print("[WARNING] Install with: pip install quantum-memory-graph")
    print("[WARNING] Middleware will run without memory (pass-through mode).")

    _fallback: list[dict] = []

    def store(text: str, source: Optional[str] = None) -> None:
        _fallback.append({"text": text, "source": source or "unknown"})

    def recall(query: str, K: int = 4) -> dict:
        recent = _fallback[-K:] if _fallback else []
        return {"memories": recent[::-1]}


# ---------------------------------------------------------------------------
# Flask import — graceful fallback
# ---------------------------------------------------------------------------
try:
    from flask import Flask, request, jsonify, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("[ERROR] Flask not installed. Install with: pip install flask")
    print("[ERROR] Cannot start middleware without Flask.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Requests import — graceful fallback
# ---------------------------------------------------------------------------
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[ERROR] requests not installed. Install with: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MIDDLEWARE_PORT = int(os.environ.get("QMG_MIDDLEWARE_PORT", "11435"))
MEMORY_K = int(os.environ.get("QMG_MEMORY_K", "4"))

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helper: inject QMG memories into the message list
# ---------------------------------------------------------------------------
def augment_messages(messages: list[dict], k: int = MEMORY_K) -> list[dict]:
    """
    Given a list of chat messages, extract the last user message,
    recall relevant QMG memories, and prepend them as a system message.

    Returns the augmented message list.
    """
    if not QMG_AVAILABLE:
        return messages  # pass-through, no augmentation

    # Find the last user message to use as the recall query
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return messages

    query = user_msgs[-1]

    try:
        result = recall(query, K=k)
        memories = result.get("memories", [])
    except Exception as e:
        print(f"[WARN] Recall failed: {e}")
        memories = []

    if not memories:
        return messages

    # Build memory context and inject as the first system message
    memory_lines = [f"- {m['text']}" for m in memories]
    memory_context = "\n".join(memory_lines)

    system_content = (
        "You are a helpful assistant with access to past conversation context. "
        "Below are relevant memories from previous exchanges. "
        "Use them to provide informed, personalized responses.\n\n"
        f"Relevant past context:\n{memory_context}"
    )

    # Prepend as system message (before any existing system messages)
    return [{"role": "system", "content": system_content}] + messages


def store_exchange(user_content: str, assistant_content: str) -> None:
    """Store a user-assistant exchange pair in QMG."""
    if not QMG_AVAILABLE or not user_content or not assistant_content:
        return
    try:
        store(f"User: {user_content}")
        store(f"Assistant: {assistant_content}")
    except Exception as e:
        print(f"[WARN] Store failed: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Handle chat completions — both streaming and non-streaming.

    Augments the request with QMG memory context, forwards to Ollama,
    and stores the completed exchange.
    """
    try:
        data = request.json
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if data is None:
        return jsonify({"error": "Request body required"}), 400

    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "messages array required"}), 400

    # ── Augment with QMG memories ──────────────────────────────────
    augmented_messages = augment_messages(messages)

    # Find the last user message for later storage
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    last_user = user_msgs[-1] if user_msgs else ""

    # ── Forward to Ollama ──────────────────────────────────────────
    data["messages"] = augmented_messages
    stream = data.get("stream", False)

    if stream:
        # Streaming path: proxy the SSE stream, collect full response, store at end
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=data,
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            return jsonify({"error": f"Cannot connect to Ollama at {OLLAMA_HOST}. Is it running?"}), 502
        except requests.exceptions.Timeout:
            return jsonify({"error": "Ollama request timed out"}), 504
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Ollama error: {str(e)}"}), 502

        def generate():
            full_response = ""
            for line in resp.iter_lines():
                if line:
                    yield line + b"\n"
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        full_response += chunk.get("message", {}).get("content", "")
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
            # Store after all chunks are sent
            if last_user and full_response:
                store_exchange(last_user, full_response)

        return Response(generate(), content_type="application/x-ndjson")

    else:
        # Non-streaming path
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=data,
                timeout=300,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            return jsonify({"error": f"Cannot connect to Ollama at {OLLAMA_HOST}. Is it running?"}), 502
        except requests.exceptions.Timeout:
            return jsonify({"error": "Ollama request timed out"}), 504
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Ollama error: {str(e)}"}), 502

        result = resp.json()

        assistant_content = result.get("message", {}).get("content", "")
        if last_user and assistant_content:
            store_exchange(last_user, assistant_content)

        return jsonify(result)


# ---------------------------------------------------------------------------
# Health / status endpoint
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    """Health check — returns middleware and upstream status."""
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_HOST}/", timeout=5)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    return jsonify({
        "status": "ok",
        "qmg_available": QMG_AVAILABLE,
        "ollama_upstream": OLLAMA_HOST,
        "ollama_reachable": ollama_ok,
        "memory_k": MEMORY_K,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  QMG Memory Middleware for Ollama")
    print("=" * 60)
    print(f"  Listening on  : http://0.0.0.0:{MIDDLEWARE_PORT}")
    print(f"  Upstream      : {OLLAMA_HOST}")
    print(f"  Memory K      : {MEMORY_K}")
    print(f"  QMG available : {QMG_AVAILABLE}")
    print(f"  Streaming     : supported")
    print()
    print("  Point your Ollama client at this server instead of :11434")
    print(f"  Example: curl http://localhost:{MIDDLEWARE_PORT}/api/chat -d '{{\"model\":\"llama3.2:3b\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'")
    print("=" * 60)

    try:
        app.run(host="0.0.0.0", port=MIDDLEWARE_PORT, debug=False)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Middleware stopped.")
