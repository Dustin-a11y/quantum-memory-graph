# QMG Integration Examples

Practical examples showing how to integrate Quantum Memory Graph with popular LLM tools.

## Examples

### 1. llama-cpp + QMG Memory (`llama_cpp_qmg_memory.py`)

Self-contained Python example that adds persistent, relationship-aware memory to
llama-cpp-python chat sessions.

**Features:**
- Injects recalled QMG memories into the system prompt before each generation
- Stores user-assistant exchanges as connected graph nodes after each turn
- Works with mock mode (no real LLM needed) for testing and demos
- Graceful fallback when `quantum-memory-graph` or `llama-cpp-python` aren't installed

**Quick start:**

```bash
# Install dependencies
pip install quantum-memory-graph llama-cpp-python

# Interactive mode (real LLM)
python llama_cpp_qmg_memory.py

# Mock mode (no model download needed, test the integration)
python llama_cpp_qmg_memory.py --mock

# Pre-recorded demo
python llama_cpp_qmg_memory.py --demo
```

**How it works:**

```
User Input → QMG recall(K memories) → Build system prompt → llama-cpp generate → QMG store(exchange) → Response
```

### 2. Ollama + QMG Middleware (`ollama_qmg_middleware.py`)

Transparent HTTP proxy that sits between any Ollama client and the Ollama server,
automatically adding QMG memory to every chat request.

**Features:**
- Intercepts `/api/chat` requests, injects recalled memories as system context
- Forwards augmented requests to Ollama, stores completed exchanges in QMG
- Handles both streaming (SSE) and non-streaming responses
- Includes `/health` endpoint for monitoring
- Graceful pass-through mode when QMG isn't installed

**Quick start:**

```bash
# Install dependencies
pip install quantum-memory-graph flask requests

# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start the middleware
python ollama_qmg_middleware.py

# Terminal 3: Use it
curl http://localhost:11435/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "What do you remember about me?"}]
}'

# Or with Ollama CLI
OLLAMA_HOST=http://localhost:11435 ollama run llama3.2:3b
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Upstream Ollama server |
| `QMG_MIDDLEWARE_PORT` | `11435` | Port the middleware listens on |
| `QMG_MEMORY_K` | `4` | Number of memories to recall per request |

**Architecture:**

```
Client → Middleware (:11435) → Ollama (:11434)
              ↕
         QMG Graph
```

## Requirements

| Example | Dependencies |
|---------|-------------|
| llama-cpp | `quantum-memory-graph`, `llama-cpp-python` |
| Ollama middleware | `quantum-memory-graph`, `flask`, `requests` |

All examples gracefully handle missing dependencies — they'll run in mock/pass-through mode with clear warnings.

## More Information

- [QMG on PyPI](https://pypi.org/project/quantum-memory-graph/)
- [QMG GitHub](https://github.com/Dustin-a11y/quantum-memory-graph)
- [QMG vs mem0 Comparison](../COMPARISON.md)
