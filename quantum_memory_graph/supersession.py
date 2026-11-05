"""State-aware recall: supersession for Kong Graph.

The graph already stores memories with timestamps, and the typed semantic layer already carries
`valid_from` / `valid_to`. Retrieval ignores all of it: `hybrid_search()` scores text + embedding,
so a fact that stopped being true two years ago ranks exactly like the one that replaced it. Ask
"where does Dustin live now" against a store holding Seattle, then Portland, then Austin, and the
answer is a coin flip between three strings that differ by one token.

This module adds the missing half — the state of an entity, not a pile of sentences about it:

  * claims     — (subject, attribute, value, when, memory_id) parsed from typed facts where present,
                 otherwise from the memory text
  * supersession — within one (subject, attribute), every claim except the newest is marked
                 superseded, with `valid_to` set to its replacement's `valid_from`
  * intent     — "where does X live now" wants the current claim; "where did X live before" wants the
                 chain. Same store, different question, different answer
  * state_aware_search() — hybrid_search + supersession annotation, so a caller can either filter
                 stale claims out or show them as history

Nothing is deleted or rewritten. Supersession is derived, so a store that never used it can start
using it today with no migration and no data loss.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# "Dustin's favorite editor is Neovim." -> subject, attribute, value
_POSSESSIVE = re.compile(
    r"^(?P<subject>[A-Za-z][^,.!?]{0,60}?)['’]s\s+(?P<attribute>[A-Za-z][A-Za-z0-9 _\-]{1,40}?)\s+"
    r"(?:is|was|are|were|will be|has become|became)\s+(?P<value>.+?)\.?$")

# "Dustin lives in Seattle." / "The rack moved to the garage." -> verb implies the attribute
_VERBAL = [
    (re.compile(r"^(?P<subject>[A-Za-z][^,.!?]{0,60}?)\s+(?:now\s+)?lives? in\s+(?P<value>.+?)\.?$", re.I),
     "lives in"),
    (re.compile(r"^(?P<subject>[A-Za-z][^,.!?]{0,60}?)\s+moved to\s+(?P<value>.+?)\.?$", re.I), "lives in"),
    (re.compile(r"^(?P<subject>[A-Za-z][^,.!?]{0,60}?)\s+works? at\s+(?P<value>.+?)\.?$", re.I), "works at"),
    (re.compile(r"^(?P<subject>[A-Za-z][^,.!?]{0,60}?)\s+prefers?\s+(?P<value>.+?)\.?$", re.I), "prefers"),
    (re.compile(r"^(?P<subject>[A-Za-z][^,.!?]{0,60}?)\s+uses?\s+(?P<value>.+?)\.?$", re.I), "uses"),
    (re.compile(r"^(?P<subject>[A-Za-z][^,.!?]{0,60}?)\s+owns?\s+(?P<value>.+?)\.?$", re.I), "owns"),
]

# "the alpha server is located in Seattle." / "the node was moved to Austin." -> attribute + value
_COPULA = re.compile(
    r"^(?P<subject>[A-Za-z][^,.!?]{0,60}?)\s+(?:is|was|are|were)\s+(?:now\s+|still\s+)?"
    r"(?P<attribute>[a-z][a-z\-]{2,20}(?:\s+[a-z][a-z\-]{2,20})?)\s+"
    r"(?:in|at|to|on|from|with)\s+(?P<value>.+?)\.?$")

CURRENT_CUES = ("now", "currently", "current", "latest", "most recent", "these days", "today",
                "still", "at the moment", "right now", "as of")
HISTORY_CUES = ("before", "previously", "used to", "originally", "first", "earlier", "in the past",
                "at the time", "when did", "history", "changed from", "prior")


def _norm(text) -> str:
    """Lowercase, collapse whitespace, and drop leading articles so "the rack", "The rack" and a
    doubled article produced by a sloppy sentence all key the same entity."""
    value = re.sub(r"\s+", " ", (text or "").strip().lower()).strip(" .?!")
    while True:
        stripped = re.sub(r"^(?:the|a|an)\s+", "", value)
        if stripped == value:
            return value
        value = stripped


@dataclass
class Claim:
    subject: str
    attribute: str
    value: str
    when: Optional[datetime]
    memory_id: str
    text: str
    source: str = ""
    superseded_by: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

    @property
    def is_current(self) -> bool:
        return self.superseded_by is None

    def as_dict(self) -> Dict:
        return {
            "subject": self.subject, "attribute": self.attribute, "value": self.value,
            "when": self.when.isoformat() if self.when else None,
            "memory_id": self.memory_id, "text": self.text, "superseded_by": self.superseded_by,
            "is_current": self.is_current,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
        }


def parse_claim(text: str, memory_id: str = "", when: Optional[datetime] = None,
                source: str = "") -> Optional[Claim]:
    """Extract a single (subject, attribute, value) claim from one memory's text, or None.

    Deliberately conservative: a sentence that does not look like a state assertion yields nothing,
    because a false claim is worse than no claim — it would supersede a real one.
    """
    body = (text or "").strip()
    if not body or len(body) > 400:
        return None
    m = _POSSESSIVE.match(body)
    if m:
        subject, attribute, value = m.group("subject"), m.group("attribute"), m.group("value")
    elif _COPULA.match(body):
        m = _COPULA.match(body)
        subject, attribute, value = m.group("subject"), m.group("attribute"), m.group("value")
    else:
        subject = attribute = value = None
        for pattern, implied in _VERBAL:
            m = pattern.match(body)
            if m:
                subject, attribute, value = m.group("subject"), implied, m.group("value")
                break
        if subject is None:
            return None
    subject, attribute, value = _norm(subject), _norm(attribute), (value or "").strip(" .")
    if not subject or not attribute or not value:
        return None
    return Claim(subject=subject, attribute=attribute, value=value, when=when,
                 memory_id=memory_id, text=body, source=source,
                 valid_from=when)


class StateIndex:
    """Supersession index over a set of claims: what is true now, and what came before."""

    def __init__(self, claims: Optional[List[Claim]] = None):
        self.claims: List[Claim] = claims or []
        self._by_key: Dict[Tuple[str, str], List[Claim]] = {}
        self._by_memory: Dict[str, Claim] = {}
        self._build()

    # ---------- construction ----------

    @classmethod
    def from_graph(cls, graph) -> "StateIndex":
        claims: List[Claim] = []
        for mem_id, mem in getattr(graph, "memories", {}).items():
            claim = parse_claim(getattr(mem, "text", ""), mem_id,
                                getattr(mem, "timestamp", None), getattr(mem, "source", ""))
            if claim:
                claims.append(claim)
        return cls(claims)

    @classmethod
    def from_typed_facts(cls, facts: List[Dict]) -> "StateIndex":
        """Authoritative path: the semantic layer already carries subject/relation/value/validity."""
        claims = []
        for f in facts:
            when = f.get("valid_from") or f.get("observed_at")
            if isinstance(when, str):
                try:
                    when = datetime.fromisoformat(when.replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    when = None
            claims.append(Claim(subject=_norm(str(f.get("subject_id") or f.get("subject_label") or "")),
                                attribute=_norm(str(f.get("relation_type") or "")),
                                value=str(f.get("value") or ""),
                                when=when, memory_id=str(f.get("fact_id") or f.get("source_memory_id") or ""),
                                text=str(f.get("value") or ""), source=str(f.get("source") or ""),
                                valid_from=when))
        return cls([c for c in claims if c.subject and c.attribute and c.value])

    def _build(self) -> None:
        for claim in self.claims:
            self._by_key.setdefault((claim.subject, claim.attribute), []).append(claim)
            if claim.memory_id:
                self._by_memory[claim.memory_id] = claim
        for chain in self._by_key.values():
            chain.sort(key=lambda c: (c.when or datetime.min))
            for older, newer in zip(chain, chain[1:]):
                if _norm(older.value) == _norm(newer.value):
                    # A restatement, not a change: keep both current, link nothing.
                    continue
                older.superseded_by = newer.memory_id or newer.value
                older.valid_to = newer.valid_from

    # ---------- queries ----------

    def history(self, subject: str, attribute: str) -> List[Claim]:
        return list(self._by_key.get((_norm(subject), _norm(attribute)), ()))

    def current(self, subject: str, attribute: str) -> Optional[Claim]:
        chain = self.history(subject, attribute)
        return chain[-1] if chain else None

    def is_superseded(self, memory_id: str) -> bool:
        claim = self._by_memory.get(memory_id)
        return bool(claim and not claim.is_current)

    def annotate(self, memory_id: str) -> Dict:
        claim = self._by_memory.get(memory_id)
        if not claim:
            return {"tracked": False}
        current = self.current(claim.subject, claim.attribute)
        return {"tracked": True, "subject": claim.subject, "attribute": claim.attribute,
                "value": claim.value, "is_current": claim.is_current,
                "superseded_by": claim.superseded_by,
                "valid_to": claim.valid_to.isoformat() if claim.valid_to else None,
                "current_value": current.value if current else None,
                "history_depth": len(self.history(claim.subject, claim.attribute))}

    def _asked_attribute(self, query: str) -> Optional[str]:
        """Which attribute is the question about? Matched against the attributes the store knows."""
        q_tokens = set(re.findall(r"[a-z0-9]+", _norm(query)))
        best, best_score = None, 0
        for (_, attribute) in self._by_key:
            for word in attribute.split():
                stem = word[:-1] if len(word) > 3 and word.endswith("s") else word
                if len(stem) >= 4 and stem in q_tokens and len(stem) > best_score:
                    best, best_score = attribute, len(stem)
        return best

    def _subject_in(self, query: str) -> Optional[str]:
        q = _norm(query)
        best = None
        for (subject, _) in self._by_key:
            if subject and subject in q and (best is None or len(subject) > len(best)):
                best = subject
        return best

    def resolve(self, query: str) -> Dict:
        """Find the entity and attribute a question is about, then say what state it is in."""
        q = _norm(query)
        intent = "current"
        if any(cue in q for cue in HISTORY_CUES) and not any(cue in q for cue in CURRENT_CUES):
            intent = "history"

        subject = self._subject_in(query)
        asked = self._asked_attribute(query)
        if subject is None:
            return {"intent": intent, "matched": False}

        chain = self.history(subject, asked) if asked else []
        hops = 0
        # A pointer ("the replica's primary is the main server") carries no value of its own: follow
        # it and ask the referent the same question. Without this a replica question is answerable
        # only by accident.
        while not chain and hops < 3:
            pointers = [c for (s, _), cs in self._by_key.items() if s == subject for c in cs]
            referent = next((_norm(c.value) for c in pointers
                             if (_norm(c.value), asked or "") in self._by_key), None)
            if referent is None:
                break
            subject, chain, hops = referent, self.history(referent, asked) if asked else [], hops + 1
        if not chain and asked:
            chain = self.history(subject, asked)
        if not chain:
            return {"intent": intent, "matched": False}
        current = chain[-1]
        return {"intent": intent, "matched": True, "subject": subject, "attribute": asked,
                "current": current.as_dict(), "hops": hops,
                "previous": chain[-2].as_dict() if len(chain) > 1 else None,
                "history": [c.as_dict() for c in chain]}


def state_aware_search(query: str, graph, k: int = 5, text_weight: float = 0.3,
                       embed_weight: float = 0.7, index: Optional[StateIndex] = None) -> List[Dict]:
    """`hybrid_search` plus supersession.

    Every hit is annotated with its state, and when the store can resolve the questioned attribute
    the right side of the change is guaranteed to be in the top slot: the live claim for a "now"
    question, the replaced value for a "before" question. Retrieval still ranks; it just stops
    presenting expired facts as if they were current.
    """
    from .search import hybrid_search

    index = index or StateIndex.from_graph(graph)
    results = hybrid_search(query=query, graph=graph, k=k,
                            text_weight=text_weight, embed_weight=embed_weight)
    resolved = index.resolve(query)
    for r in results:
        r["state"] = index.annotate(r["id"])
    if not resolved.get("matched"):
        return results

    target = resolved["current"] if resolved.get("intent") == "current" else resolved.get("previous")
    if not target:
        return results
    wanted = (target.get("value") or "").lower()
    if results and wanted and wanted in (results[0].get("text") or "").lower():
        return results
    results.sort(key=lambda r: (not r["state"].get("is_current", False)))
    results.insert(0, {
        "id": target.get("memory_id"), "text": target.get("text") or _claim_sentence(resolved),
        "entities": [], "source": "supersession-resolution",
        "text_score": 0.0, "embed_score": 0.0, "combined_score": 1.0,
        "state": index.annotate(target.get("memory_id") or ""),
        "why": "%s (%s) — %s" % (resolved["subject"], resolved["attribute"], resolved["intent"]),
    })
    return results[:max(k, 1)]


def _claim_sentence(resolved: Dict) -> str:
    c = resolved["current"]
    return "%s's %s is %s." % (c["subject"], c["attribute"], c["value"])
