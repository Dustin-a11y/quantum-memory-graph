"""QMG Memory Provider — automatic turn-level memory sync to Quantum Memory Graph.

Classifies each turn as EPHEMERAL / DURABLE / CRITICAL and stores only
DURABLE + CRITICAL turns to the agent's personal QMG graph. No agent
decision required — fully automatic.

Config via $HERMES_HOME/qmg-memory/config.json (profile-scoped):
  {
    "qmg_url": "http://127.0.0.1:8514",
    "agent_name": "toadstool",
    "store_critical": true,
    "store_durable": true,
    "store_ephemeral": false,
    "min_turn_length": 10,
    "max_memory_length": 500,
    "extraction_model": "deepseek/deepseek-chat",
    "enabled": true
  }
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home


logger = logging.getLogger(__name__)

# ── Classification patterns ──────────────────────────────────────────────

# EPHEMERAL: process noise, negotiation, filler — never store
_EPHEMERAL_PATTERNS = [
    r"^(ok|okay|sure|yes|no|yeah|nah|maybe|thanks|thank you|thx|ty)\b",
    r"^(got it|understood|makes sense|i see|right|correct|wrong)\b",
    r"^(let me|i'll|i will|i can|i should|let's)\b.*(try|check|look|see|verify)",
    r"^(actually|wait|hold on|hmm|umm|uh)\b",
    r"^(what|how|when|where|why|which|who)\b.*\?$",
    r"^(can you|could you|would you|will you)\b.*\?$",
    r"^(is that|does that|do you|are you)\b.*\?$",
    r"^(i mean|i meant|sorry|my bad|oops)\b",
    r"^(never mind|forget it|ignore that|disregard)\b",
    r"^(lol|haha|lmao|nice|cool|awesome|great|amazing)\b",
    r"^\s*$",
    r"^\[.*\]$",  # pure metadata tags
]

# DURABLE: facts, preferences, decisions, configs — store
_DURABLE_PATTERNS = [
    r"(?:i|we|the team|the fleet)\s+(?:prefer|like|want|need|decided|chose|picked|went with|use|using|switched to|moved to|migrated to)\b",
    r"(?:port|ip|url|path|directory|file|config|setting|credential|token|key|secret)\s+(?:is|=|:|uses?|runs? on|at)\s*\S+",
    r"(?:uses?|runs? on|at|via|through)\s+(?:port|ip|url|path|directory|file|config|setting|credential|token|key|secret)\s+\S+",
    r"(?:the|this|that)\s+(?:machine|server|service|agent|profile|instance)\s+(?:is|=|:)\s*\S+",
    r"(?:version|release|tag|commit|sha|hash)\s+(?:is|=|:)\s*\S+",
    r"(?:fixed|bug|issue|problem|error|failure|crash)\s+(?:was|is|in|at|on)\b",
    r"(?:root cause|solution|fix|patch|workaround)\s+(?:was|is|:)\b",
    r"(?:deployed|shipped|released|merged|pushed|committed)\b",
    r"(?:created|built|wrote|implemented|added|removed|deleted|updated|changed)\s+(?:the|a|an)\s+\w+",
    r"(?:always|never|every|each|all|none)\s+(?:of the|agents|sessions|runs|times)",
    r"(?:qmg|memory graph|kong graph|mega brain|dual-write|benchd|canary)\b",
    r"(?:recipe|brief|ingredient|sauce|rice|slaw|chicken|bowl|pasta|pizza)\b.*(?:is|=|:|with|uses|using)",
    r"(?:monday\.com|parsley|cookunity|midjourney|chef-os|radicch\.io)\b",
]

# CRITICAL: security, architecture, credentials, user corrections — always store
_CRITICAL_PATTERNS = [
    r"(?:password|secret|api.?key|token|credential|private.?key|ssh|oauth|bearer)\s+(?:is|=|:)\s*\S+",
    r"(?:never|always|do not|don't|must|critical|important|warning|danger|caution)\b.*(?:store|save|remember|note)",
    r"(?:i was wrong|i made a mistake|my mistake|correction|corrected)\b",
    r"(?:security|vulnerability|exploit|backdoor|breach|leak|exposure)\b",
    r"(?:architecture|design decision|breaking change|migration|deprecat)\b",
    r"(?:stop|don't do that|never do that|do not do that)\b",
    r"(?:production|prod|live|staging|dev|sandbox)\s+(?:is|=|:|server|environment)",
    r"(?:user preference|preference|prefers|preferred)\s+(?:is|=|:)\s*\S+",
]

# Minimum turn length to consider (skip trivial exchanges)
_MIN_TURN_LENGTH = 10
_MAX_MEMORY_LENGTH = 500


def _classify_turn(user_content: str, assistant_content: str) -> str:
    """Classify a turn as EPHEMERAL, DURABLE, or CRITICAL."""
    combined = f"{user_content}\n{assistant_content}".lower()
    
    # Check CRITICAL first
    for pattern in _CRITICAL_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return "CRITICAL"
    
    # Check DURABLE — assistant content carries more weight
    for pattern in _DURABLE_PATTERNS:
        if re.search(pattern, assistant_content, re.IGNORECASE):
            return "DURABLE"
    
    # Also check combined for durable patterns
    for pattern in _DURABLE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return "DURABLE"
    
    # Check EPHEMERAL — but only if the assistant didn't give a substantive answer
    # A substantive answer (>20 chars with factual content) overrides ephemeral question
    is_substantive = len(assistant_content) > 20 and any(
        c.isalnum() for c in assistant_content
    )
    
    if not is_substantive:
        for pattern in _EPHEMERAL_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                return "EPHEMERAL"
    
    # Default: if assistant gave a substantive answer (>50 chars), it's DURABLE
    if is_substantive:
        return "DURABLE"
    
    return "EPHEMERAL"


# ── Kong Graph v2 LLM extraction (opt-in) ─────────────────────────────────
# Benchmark-validated upgrade. Kong-learner: v2 prompt LoCoMo entity recall 0.9066
# vs 0.8765 old / 0.8842 production; full-transcript standalone LME 0.8300 (was
# 0.7945); union(new-ft, old) deployable 0.9198 > 0.9048. Rollout 2026-09-20.
_V2_TURN_PROMPT = """You are an expert fact-extraction engine for a long-term memory system. Read ONE conversation turn (one user message and the assistant's reply) and output a COMPLETE set of atomic facts worth remembering long-term. Recall and specificity matter more than brevity.

EXTRACTION RULES:
1. COMPLETENESS: Extract every durable fact: biographical details, preferences, possessions, plans, decisions, configs, paths, ports, credential references, events, relationships, and small details. Do not summarize away "minor" details.
2. VERBATIM ENTITIES: Preserve every specific named entity exactly as written: names, machines, IPs, ports, paths, URLs, models, repos, dates, times, quantities, prices. NEVER generalize.
3. PRONOUN RESOLUTION: Every fact must be self-contained. Replace pronouns with the specific name or thing.
4. INCLUDE ASSISTANT CONTENT: The assistant's statements, decisions, results, and explanations are facts too — extract them ("The assistant confirmed ...", "The result was ...", "X is configured to ...").
5. ABSOLUTE DATES: Convert relative time ("yesterday", "last week") to absolute dates using the session date when provided.
6. ATOMIC: One piece of information per fact. Short declarative sentences.
7. FAITHFULNESS: Only facts stated or directly implied. Never invent details.

OUTPUT FORMAT: a single JSON object: {"facts": ["fact 1", "fact 2", ...]} — nothing outside the JSON object. If the turn has nothing durable, output {"facts": []}."""


def _parse_v2_facts(content):
    """Parse {"facts":[...]} from model output; tolerate code fences."""
    if not content:
        return None
    s = re.sub(r'^```(?:json)?\s*', '', content.strip())
    s = re.sub(r'\s*```$', '', s)
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        m = re.search(r'(\{.*\}|\[.*\])', s, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
            except Exception:
                return None
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k in ('facts', 'FACTS', 'data', 'list'):
            if isinstance(obj.get(k), list):
                obj = obj[k]
                break
    if isinstance(obj, list):
        facts = [x.strip() for x in obj if isinstance(x, str) and x.strip()]
        return facts or None
    return None


def _load_nous_key():
    """Read current Nous agent_key from the profile auth.json (Hermes rotates hourly)."""
    try:
        auth_path = os.path.join(get_hermes_home(), 'auth.json')
        d = json.load(open(auth_path))
        return d['providers']['nous']['agent_key']
    except Exception:
        return None


def _llm_extract_facts(user_content, assistant_content, *, model, inference_url,
                       session_date, max_facts=12, timeout=45):
    """Call an OpenAI-compatible chat endpoint to extract atomic facts from a turn.
    Returns a list of fact strings, or None on any failure (caller falls back)."""
    url = f"{inference_url.rstrip('/')}/chat/completions"
    date_line = f"Session date: {session_date}\n\n" if session_date else ""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _V2_TURN_PROMPT},
            {"role": "user", "content": date_line + f"User: {user_content}\n\nAssistant: {assistant_content}"},
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }
    last = None
    for attempt in range(3):
        key = _load_nous_key()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            r = requests.post(url, json=body, headers=headers, timeout=timeout)
            if r.status_code == 401 and attempt < 2:
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            msg = r.json()['choices'][0]['message']
            facts = _parse_v2_facts(msg.get('content') or msg.get('reasoning'))
            if facts:
                return facts[:max_facts]
            last = 'empty-parse'
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(2 * (attempt + 1))
    logger.debug("QMG LLM extraction failed (%s); using heuristic fallback", last)
    return None


def _extract_memory_facts(user_content: str, assistant_content: str, config: Dict[str, Any] = None) -> List[str]:
    """Extract memory fact(s) for a turn. Returns a list.

    If config['use_llm_extraction'] is on, use the Kong Graph v2 LLM extractor
    (deepseek via Nous Portal) which returns several atomic facts including
    assistant-content facts. On any LLM failure, falls back to the heuristic.
    """
    config = config or {}
    if config.get("use_llm_extraction"):
        facts = _llm_extract_facts(
            user_content, assistant_content,
            model=config.get("extraction_model", "deepseek/deepseek-chat-v3.1"),
            inference_url=config.get("extraction_url", "https://inference-api.nousresearch.com/v1"),
            session_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            max_facts=int(config.get("extraction_max_facts", 12)),
        )
        if facts:
            return facts
    return [_extract_memory_text(user_content, assistant_content)]


def _extract_memory_text(user_content: str, assistant_content: str) -> str:
    """Heuristic fallback: the most important sentence of the turn."""
    # Simple heuristic: find the longest sentence in assistant content
    sentences = re.split(r'[.!?]+', assistant_content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        # Fall back to user content if assistant was empty
        sentences = re.split(r'[.!?]+', user_content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return f"User: {user_content[:100]} | Assistant: {assistant_content[:100]}"
    
    # Take the longest sentence (usually the most informative)
    key_sentence = max(sentences, key=len)
    
    # If it's too long, truncate
    if len(key_sentence) > _MAX_MEMORY_LENGTH:
        key_sentence = key_sentence[:_MAX_MEMORY_LENGTH-3] + "..."
    
    return key_sentence


class QMGMemoryProvider(MemoryProvider):
    """Memory provider that auto-syncs turns to QMG with smart filtering."""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._enabled = False
        self._qmg_url = ""
        self._agent_name = ""
        self._session_id = ""
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._turn_count = 0
        self._stored_count = 0
        self._skipped_count = 0

    @property
    def name(self) -> str:
        return "qmg"

    def is_available(self) -> bool:
        """Check if QMG is reachable and config is present."""
        config_path = os.path.join(get_hermes_home(), "qmg-memory", "config.json")
        if not os.path.exists(config_path):
            return False
        
        try:
            with open(config_path) as f:
                self._config = json.load(f)
        except Exception:
            return False
        
        if not self._config.get("enabled", True):
            return False
        
        qmg_url = self._config.get("qmg_url", "")
        if not qmg_url:
            return False
        
        # Quick health check
        try:
            headers = {}
            token = self._config.get("auth_token", "")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            r = requests.get(f"{qmg_url}/", timeout=2, headers=headers)
            return r.status_code == 200
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize for a session."""
        config_path = os.path.join(get_hermes_home(), "qmg-memory", "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                self._config = json.load(f)
        
        self._enabled = self._config.get("enabled", True)
        self._qmg_url = self._config.get("qmg_url", "")
        self._agent_name = self._config.get("agent_name", kwargs.get("agent_identity", "unknown"))
        self._session_id = session_id
        
        if not self._enabled:
            logger.info("QMG memory provider disabled by config")
            return
        
        # Start background writer thread
        self._shutdown.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        
        logger.info(
            "QMG memory provider initialized for %s (url=%s, session=%s)",
            self._agent_name, self._qmg_url, session_id
        )

    def system_prompt_block(self) -> str:
        """Static text for the system prompt."""
        if not self._enabled:
            return ""
        
        return (
            f"\n## Automatic Memory (QMG)\n"
            f"Your conversations are automatically analyzed and durable facts are stored "
            f"to your Quantum Memory Graph at {self._qmg_url}. No action required from you — "
            f"the system classifies each turn and stores only what matters.\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context for the upcoming turn."""
        if not self._enabled or not self._qmg_url:
            return ""
        
        try:
            headers = {}
            token = self._config.get("auth_token", "")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            # /recall selects from a fixed 14-candidate window and ranks by graph walk, so a
            # short question-echo ("The assistant needed to know about donor laptops.") beats the
            # answer that exists in the graph, and anything outside those 14 nodes cannot surface
            # at all. /search blends embedding + lexical similarity across the whole graph, which
            # is what the recall tool uses. Injection must use the same honest scorer, and k=6 so
            # an aggregation question can carry more than three facts.
            r = requests.post(
                f"{self._qmg_url}/search",
                json={"query": query, "k": 6},
                headers=headers,
                timeout=3
            )
            if r.status_code == 200:
                data = r.json()
                memories = data.get("results") or data.get("memories", [])
                if memories:
                    lines = ["## Relevant Memories (QMG)"]
                    for mem in memories:
                        text = (mem.get("text") or mem.get("memory") or "")[:150]
                        lines.append(f"- {text}")
                    return "\n".join(lines)
        except Exception:
            pass
        
        return ""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Classify and queue a turn for storage. Non-blocking."""
        if not self._enabled:
            return
        
        self._turn_count += 1
        
        # Skip trivial turns
        if len(user_content) < _MIN_TURN_LENGTH and len(assistant_content) < _MIN_TURN_LENGTH:
            self._skipped_count += 1
            return
        
        # Classify the turn
        classification = _classify_turn(user_content, assistant_content)
        
        # Store based on classification
        should_store = False
        if classification == "CRITICAL" and self._config.get("store_critical", True):
            should_store = True
        elif classification == "DURABLE" and self._config.get("store_durable", True):
            should_store = True
        elif classification == "EPHEMERAL" and self._config.get("store_ephemeral", False):
            should_store = True
        
        if not should_store:
            self._skipped_count += 1
            logger.debug("QMG sync_turn: skipped %s turn %d", classification.lower(), self._turn_count)
            return
        
        # Extract memory fact(s) — LLM v2 if enabled, else heuristic single
        facts = _extract_memory_facts(user_content, assistant_content, self._config)
        for memory_text in facts:
            self._write_queue.put({
                "text": memory_text,
                "source": f"{self._agent_name}-auto",
                "classification": classification,
                "session_id": session_id or self._session_id,
                "turn": self._turn_count,
            })

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """No tools exposed — this is a passive provider."""
        return []

    def shutdown(self) -> None:
        """Flush queue and stop writer thread."""
        self._shutdown.set()
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5)
        
        # Log stats
        logger.info(
            "QMG memory provider shutdown: %d turns, %d stored, %d skipped",
            self._turn_count, self._stored_count, self._skipped_count
        )

    def _writer_loop(self) -> None:
        """Background writer thread — drains queue to QMG."""
        while not self._shutdown.is_set():
            try:
                item = self._write_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            try:
                headers = {}
                token = self._config.get("auth_token", "")
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                r = requests.post(
                    f"{self._qmg_url}/store",
                    json={
                        "text": item["text"],
                        "source": item["source"],
                    },
                    headers=headers,
                    timeout=5,
                )
                if r.status_code == 200:
                    self._stored_count += 1
                    logger.debug(
                        "QMG stored %s memory (turn %d): %s",
                        item["classification"], item["turn"], item["text"][:80]
                    )
                else:
                    logger.warning("QMG store failed: HTTP %s", r.status_code)
            except Exception as e:
                logger.warning("QMG store error: %s", e)
            finally:
                self._write_queue.task_done()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """End-of-session extraction — do a final pass on the full transcript."""
        if not self._enabled or not messages:
            return
        
        logger.info("QMG on_session_end: processing %d messages", len(messages))
        
        # Extract key facts from the full session using a simple heuristic
        facts = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not content:
                continue
            
            # Look for sentences with durable patterns
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20 or len(sentence) > _MAX_MEMORY_LENGTH:
                    continue
                
                for pattern in _DURABLE_PATTERNS + _CRITICAL_PATTERNS:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        facts.append(sentence)
                        break
        
        # Store unique facts
        unique_facts = list(dict.fromkeys(facts))  # dedupe preserving order
        headers = {}
        token = self._config.get("auth_token", "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        for fact in unique_facts[:10]:  # cap at 10 per session
            try:
                requests.post(
                    f"{self._qmg_url}/store",
                    json={"text": fact, "source": f"{self._agent_name}-session-end"},
                    headers=headers,
                    timeout=5,
                )
            except Exception:
                pass
        
        logger.info("QMG on_session_end: stored %d facts", len(unique_facts[:10]))


# ── Plugin registration ──────────────────────────────────────────────────

def create_provider() -> QMGMemoryProvider:
    """Factory function called by the plugin loader."""
    return QMGMemoryProvider()


def register(ctx):
    """Plugin registration hook called by the loader."""
    provider = QMGMemoryProvider()
    ctx.register_memory_provider(provider)
    return provider
