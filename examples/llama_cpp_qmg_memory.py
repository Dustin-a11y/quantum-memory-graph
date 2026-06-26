"""
llama-cpp + QMG Memory Integration
===================================
Adds persistent, relationship-aware memory to llama-cpp chat sessions.
Memories are stored in a knowledge graph; recall returns connected, relevant context.

Requirements:
    pip install quantum-memory-graph llama-cpp-python

Usage:
    python llama_cpp_qmg_memory.py              # Interactive mode (requires llama-cpp installed)
    python llama_cpp_qmg_memory.py --mock       # Mock mode (no llama-cpp needed, for testing)
    python llama_cpp_qmg_memory.py --demo       # Pre-recorded demo conversation

Integration pattern:
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  User Input   │────▶│ QMG recall() │────▶│  System      │
    │               │     │ (K relevant  │     │  Prompt      │
    └───────────────┘     │  memories)   │     └──────┬───────┘
                          └──────────────┘            │
                                                      ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Return       │◀────│ QMG store()  │◀────│ llama-cpp    │
    │  Response     │     │ (persist     │     │ generate()   │
    │               │     │  exchange)   │     │              │
    └───────────────┘     └──────────────┘     └──────────────┘

Each turn:
  1. Recall the K most relevant past memories using embedding + graph traversal + QAOA
  2. Inject those memories as context into the system prompt
  3. Generate response with llama-cpp
  4. Store the user message and assistant reply as new graph nodes
"""

import sys
import os
from typing import Optional

# ---------------------------------------------------------------------------
# QMG import — gracefully handle missing package
# ---------------------------------------------------------------------------
try:
    from quantum_memory_graph import store, recall
    QMG_AVAILABLE = True
except ImportError:
    QMG_AVAILABLE = False
    print("[WARNING] quantum-memory-graph not installed. Install with: pip install quantum-memory-graph")
    print("[WARNING] Running in degraded mode — memory will be ephemeral dict.")

    # Fallback: in-memory dict so the example still runs and demonstrates the pattern
    _fallback_memories: list[dict] = []

    def store(text: str, source: Optional[str] = None) -> None:
        """Fallback store — in-memory list, no graph relationships."""
        _fallback_memories.append({"text": text, "source": source or "unknown", "connections": []})

    def recall(query: str, K: int = 4) -> dict:
        """Fallback recall — returns most recent memories (no real search)."""
        recent = _fallback_memories[-K:] if _fallback_memories else []
        return {"memories": recent[::-1]}  # newest last in list


# ---------------------------------------------------------------------------
# llama-cpp import — handle missing package with mock mode
# ---------------------------------------------------------------------------
LLAMA_AVAILABLE = False
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    print("[INFO] llama-cpp-python not installed. Use --mock for demo without real LLM.")


# ---------------------------------------------------------------------------
# Chat session with QMG memory
# ---------------------------------------------------------------------------

class QMGMemoryChat:
    """
    A chat session that uses QMG as its persistent memory backend.

    Each exchange (user message + assistant response) is stored as two
    connected memory nodes in the knowledge graph. Before every response,
    the graph is queried for the K most relevant past memories, which are
    injected into the system prompt so the LLM has access to conversation
    history that is both relevant and relationship-aware.

    Parameters
    ----------
    model_repo : str
        HuggingFace repo ID for the GGUF model.
    model_file : str
        Filename within the repo (e.g. "Llama-3.2-3B-Instruct-Q4_K_M.gguf").
    n_ctx : int
        Context window size (default 4096).
    mock : bool
        If True, use a mock LLM that echoes back a canned response.
        Useful for testing the QMG integration without a GPU / large download.
    memory_k : int
        Number of memories to recall per turn (default 4).
    verbose : bool
        Print debug info (recalled memories, etc.).
    """

    def __init__(
        self,
        model_repo: str = "bartowski/Llama-3.2-3B-Instruct-GGUF",
        model_file: str = "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        n_ctx: int = 4096,
        mock: bool = False,
        memory_k: int = 4,
        verbose: bool = False,
    ):
        self.mock = mock
        self.memory_k = memory_k
        self.verbose = verbose
        self.llm = None

        if mock or not LLAMA_AVAILABLE:
            print("[MOCK] Using mock LLM — no real model loaded.")
        else:
            try:
                self.llm = Llama.from_pretrained(
                    repo_id=model_repo,
                    filename=model_file,
                    n_ctx=n_ctx,
                    verbose=False,
                )
                print(f"[OK] Loaded {model_repo}/{model_file}")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                print("[FALLBACK] Switching to mock mode.")
                self.mock = True

    def _generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a response — real LLM or mock."""
        if self.mock or self.llm is None:
            # Mock: return a canned response that references the prompt
            return (
                "Thanks for your message! I've noted it in my memory. "
                "In a real setup, I would use the recalled context to give you "
                "a relevant, personalized response."
            )

        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                stop=["<|user|>", "<|end|>"],
                echo=False,
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return "[Error generating response]"

    def chat(self, user_message: str) -> str:
        """
        Process one turn of conversation.

        1. Recall relevant past memories from QMG
        2. Build a system prompt that includes those memories as context
        3. Generate the assistant's response
        4. Store the full exchange back into QMG for future recall
        """
        # ── Step 1: Recall ──────────────────────────────────────────
        try:
            result = recall(user_message, K=self.memory_k)
            memories = result.get("memories", [])
        except Exception as e:
            print(f"[WARN] Recall failed: {e}")
            memories = []

        memory_context = "\n".join(
            f"- {m['text']}" for m in memories
        ) if memories else "(No relevant past memories found.)"

        if self.verbose and memories:
            print(f"\n[RECALL] Found {len(memories)} relevant memories:")
            for m in memories:
                print(f"  • {m['text'][:100]}...")

        # ── Step 2: Build prompt ────────────────────────────────────
        system = (
            "You are a helpful, memory-aware assistant. "
            "Below is relevant context from past conversations. "
            "Use it to give informed, personalized responses.\n\n"
            f"Relevant context:\n{memory_context}"
        )
        prompt = f"<|system|>\n{system}\n<|user|>\n{user_message}\n<|assistant|>"

        # ── Step 3: Generate ────────────────────────────────────────
        reply = self._generate(prompt)

        # ── Step 4: Store ───────────────────────────────────────────
        try:
            store(f"User: {user_message}")
            store(f"Assistant: {reply}")
        except Exception as e:
            print(f"[WARN] Store failed: {e}")

        return reply


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

def run_demo():
    """Pre-recorded demo conversation showing memory accumulation."""
    print("=" * 60)
    print("  QMG + llama-cpp Memory Demo (mock mode)")
    print("=" * 60)

    chat = QMGMemoryChat(mock=True, memory_k=3, verbose=True)

    turns = [
        "Hi! My name is Alex and I'm building a chatbot with FastAPI.",
        "I decided to use PostgreSQL for the database.",
        "What do you remember about my project?",
        "Also, I chose React for the frontend. What's my full stack?",
    ]

    for i, msg in enumerate(turns, 1):
        print(f"\n{'─' * 40}")
        print(f"[Turn {i}] User: {msg}")
        reply = chat.chat(msg)
        print(f"[Turn {i}] Assistant: {reply}")

    print(f"\n{'─' * 40}")
    print("Demo complete. All exchanges stored in QMG memory graph.")


def run_interactive():
    """Interactive chat loop with real or mock LLM."""
    mock = "--mock" in sys.argv
    chat = QMGMemoryChat(mock=mock, memory_k=4, verbose=True)

    print("=" * 60)
    print("  QMG Memory Chat — type 'quit' to exit, 'memories' to recall")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "memories":
            result = recall("recent conversation", K=10)
            print("\n[Recent Memories]")
            for m in result.get("memories", []):
                print(f"  • {m['text']}")
            continue

        reply = chat.chat(user_input)
        print(f"Assistant: {reply}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()
