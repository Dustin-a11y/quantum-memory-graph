"""Opt-in typed semantic facts for the isolated QMG Layer 3 sandbox.

This module never extracts facts from text or chooses a winning assertion. Callers
must supply explicit entity IDs, provenance, and an aware observation time. Facts
are indexed separately from the legacy text/embedding graph, so conflicting
observations remain available for a later, explicit conflict-review layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Relation names are deliberately closed, not arbitrary caller-supplied labels.
# Relation prefix denotes the required subject entity type.
RELATION_TYPES = {
    "user.prefers": ("user", frozenset({"string"})),
    "project.status": ("project", frozenset({"string"})),
    "project.owns": ("project", frozenset({"entity_ref"})),
    "config.value": ("config", frozenset({"string", "integer", "number", "boolean"})),
    "api.located_at": ("api", frozenset({"string"})),
    "service.uses": ("service", frozenset({"entity_ref"})),
}
_ENTITY_TYPES = frozenset({"user", "project", "config", "api", "service", "repo", "store"})
_ENTITY_ID = re.compile(r"^([a-z][a-z0-9_-]*):([A-Za-z0-9][A-Za-z0-9._:/-]*)$")
_ISO_AWARE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)
_FACT_FIELDS = frozenset({
    "subject_id", "subject_label", "relation_type", "value", "datatype",
    "source", "source_memory_id", "observed_at", "valid_from", "valid_to",
    "confidence", "provenance",
})


class AmbiguousEntityError(ValueError):
    """A display label identifies more than one explicit entity ID."""


def _entity_type(entity_id: str) -> str:
    if not isinstance(entity_id, str):
        raise ValueError("entity ID must be a namespaced string")
    match = _ENTITY_ID.fullmatch(entity_id)
    if match is None or match.group(1) not in _ENTITY_TYPES:
        raise ValueError("entity ID must use an allowlisted type:key namespace")
    return match.group(1)


def _validate_subject_relation(subject_id: str, relation_type: str) -> None:
    if not isinstance(relation_type, str) or relation_type not in RELATION_TYPES:
        raise ValueError("relation_type is not allowlisted")
    if _entity_type(subject_id) != RELATION_TYPES[relation_type][0]:
        raise ValueError("subject_id namespace does not match relation_type")


def _utc(value: Any, name: str, *, required: bool = False) -> Optional[datetime]:
    if value is None:
        if required:
            raise ValueError(f"{name} requires an explicit UTC-aware timestamp")
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and _ISO_AWARE.fullmatch(value):
        try:
            parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid ISO8601 timestamp") from exc
    else:
        raise ValueError(f"{name} requires an aware datetime or strict ISO8601 string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} cannot be timezone-naive")
    return parsed.astimezone(timezone.utc)


def _validate_value(datatype: str, value: Any) -> Any:
    if datatype == "string":
        if type(value) is not str or not value.strip():
            raise ValueError("string fact value must be nonblank text")
    elif datatype == "entity_ref":
        _entity_type(value)
    elif datatype == "integer":
        if type(value) is not int:
            raise ValueError("integer fact value must be an integer, not bool")
    elif datatype == "number":
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("number fact value must be finite and not bool")
    elif datatype == "boolean":
        if type(value) is not bool:
            raise ValueError("boolean fact value must be bool")
    else:
        raise ValueError("datatype is not allowlisted")
    return value


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat(timespec="microseconds") if value is not None else None



def _require_utc_stored(value: Any, name: str, *,
                        required: bool = False) -> Optional[datetime]:
    """Persisted-state dates: real, timezone-aware, at exactly UTC offset zero.

    A +02:00 datetime is the same instant as its UTC twin, but it is not the
    canonical persisted form, so it must fail closed rather than be silently
    normalized. Only the write path normalizes formats.
    """
    if value is None:
        if required:
            raise ValueError(f"{name} requires an explicit UTC-aware timestamp")
        return None
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a stored datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} cannot be timezone-naive")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be stored at UTC offset zero")
    return value


def _canonical_payload(subject_id: str, relation_type: str, value: Any,
                       datatype: str, source: str, provenance: Optional[str],
                       observed_at: datetime, valid_from: Optional[datetime],
                       valid_to: Optional[datetime],
                       confidence: Optional[float]) -> Dict[str, Any]:
    """The exact schema-v1 canonical JSON payload the fact ID commits to."""
    return {
        "schema": "qmg-semantic-fact-v1",
        "subject_id": subject_id,
        "relation_type": relation_type,
        "value": value,
        "datatype": datatype,
        "source": source,
        "provenance": provenance,
        "observed_at": _iso(observed_at),
        "valid_from": _iso(valid_from),
        "valid_to": _iso(valid_to),
        "confidence": confidence,
    }


def _derive_fact_id(subject_id: str, relation_type: str, value: Any,
                    datatype: str, source: str, provenance: Optional[str],
                    observed_at: datetime, valid_from: Optional[datetime],
                    valid_to: Optional[datetime],
                    confidence: Optional[float]) -> str:
    """SHA-256 of the canonical v1 payload; byte-identical to the write path."""
    payload = json.dumps(
        _canonical_payload(subject_id, relation_type, value, datatype, source,
                           provenance, observed_at, valid_from, valid_to,
                           confidence),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_canonical_fact(*, subject_id: str, relation_type: str, value: Any,
                             datatype: str, source: str,
                             subject_label: Optional[str],
                             provenance: Optional[str], observed_at: Any,
                             valid_from: Any, valid_to: Any,
                             confidence: Optional[float],
                             persisted: bool) -> Dict[str, Any]:
    """One canonical schema-v1 validator shared by the write and load paths.

    ``persisted=False`` (write path) accepts aware datetimes or strict ISO8601
    strings and normalizes them to UTC. ``persisted=True`` (load path) requires
    dates already stored as aware datetimes at UTC offset zero; malformed
    persisted state is rejected, never normalized.
    """
    _validate_subject_relation(subject_id, relation_type)
    if not isinstance(datatype, str) or datatype not in RELATION_TYPES[relation_type][1]:
        raise ValueError("datatype is not permitted for relation_type")
    _validate_value(datatype, value)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("typed fact requires a nonblank source")
    if subject_label is not None:
        if not isinstance(subject_label, str) or not subject_label.strip():
            raise ValueError("subject_label must be nonblank text")
        normalized_label = subject_label.strip()
        if persisted and subject_label != normalized_label:
            raise ValueError("stored subject_label is not canonical")
        subject_label = normalized_label
    if provenance is not None and (not isinstance(provenance, str) or not provenance.strip()):
        raise ValueError("provenance must be nonblank text when supplied")
    if confidence is not None and (
        type(confidence) is not float or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError("confidence must be a finite float in [0, 1]")
    if persisted:
        observed_at = _require_utc_stored(observed_at, "observed_at", required=True)
        valid_from = _require_utc_stored(valid_from, "valid_from")
        valid_to = _require_utc_stored(valid_to, "valid_to")
    else:
        observed_at = _utc(observed_at, "observed_at", required=True)
        valid_from = _utc(valid_from, "valid_from")
        valid_to = _utc(valid_to, "valid_to")
    if valid_from is not None and valid_to is not None and valid_from >= valid_to:
        raise ValueError("valid_from must precede valid_to")
    return {
        "subject_id": subject_id,
        "relation_type": relation_type,
        "value": value,
        "datatype": datatype,
        "source": source,
        "subject_label": subject_label,
        "provenance": provenance,
        "observed_at": observed_at,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "confidence": confidence,
    }


@dataclass(frozen=True)
class SemanticFact:
    """A single assertion, not a current-value/supersession decision."""

    id: str
    subject_id: str
    relation_type: str
    value: Any
    datatype: str
    source: str
    source_memory_id: str
    observed_at: datetime
    subject_label: Optional[str] = None
    provenance: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    confidence: Optional[float] = None
    evidence_memory_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe representation with every provenance and time field."""
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "subject_label": self.subject_label,
            "relation_type": self.relation_type,
            "value": self.value,
            "datatype": self.datatype,
            "source": self.source,
            "source_memory_id": self.source_memory_id,
            "evidence_memory_ids": list(self.evidence_memory_ids or (self.source_memory_id,)),
            "provenance": self.provenance,
            "observed_at": _iso(self.observed_at),
            "valid_from": _iso(self.valid_from),
            "valid_to": _iso(self.valid_to),
            "confidence": self.confidence,
        }


# Schema-v1 pickle writes carry every dataclass field, including optional
# fields explicitly set to None. Check __dict__, not getattr: dataclass class
# defaults otherwise mask fields deleted from a corrupted pickle.
_FACT_V1_ATTRS = frozenset(SemanticFact.__dataclass_fields__)


class SemanticStore:
    """Exact subject+relation index, explicit entity aliases, and full versions."""

    def __init__(self):
        self.facts: Dict[str, SemanticFact] = {}
        self.by_subject_relation: Dict[Tuple[str, str], set] = {}
        self.labels_by_id: Dict[str, set] = {}
        self.label_index: Dict[str, set] = {}

    @staticmethod
    def prepare(raw_facts: Iterable[dict], source: str,
                source_memory_id: str) -> List[SemanticFact]:
        """Validate the WHOLE write before the legacy graph is mutated."""
        if not isinstance(raw_facts, (list, tuple)) or not raw_facts:
            raise ValueError("semantic_facts must be a nonempty list")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("typed facts require a nonblank memory source")
        if not isinstance(source_memory_id, str) or not source_memory_id.strip():
            raise ValueError("typed facts require a nonblank source_memory_id")
        prepared = []
        for raw in raw_facts:
            if not isinstance(raw, dict):
                raise ValueError("each semantic fact must be a mapping")
            unknown = set(raw) - (_FACT_FIELDS - {"source", "source_memory_id"})
            if unknown:
                raise ValueError(f"unknown or spoofed fact fields: {sorted(unknown)}")
            required = {"subject_id", "relation_type", "value", "datatype", "observed_at"}
            if not required.issubset(raw):
                raise ValueError(f"missing fact fields: {sorted(required - set(raw))}")
            fields = _validate_canonical_fact(
                subject_id=raw["subject_id"],
                relation_type=raw["relation_type"],
                value=raw["value"],
                datatype=raw["datatype"],
                source=source,
                subject_label=raw.get("subject_label"),
                provenance=raw.get("provenance"),
                observed_at=raw["observed_at"],
                valid_from=raw.get("valid_from"),
                valid_to=raw.get("valid_to"),
                confidence=raw.get("confidence"),
                persisted=False,
            )
            fact_id = _derive_fact_id(
                fields["subject_id"], fields["relation_type"], fields["value"],
                fields["datatype"], fields["source"], fields["provenance"],
                fields["observed_at"], fields["valid_from"], fields["valid_to"],
                fields["confidence"],
            )
            prepared.append(SemanticFact(
                id=fact_id, subject_id=fields["subject_id"],
                subject_label=fields["subject_label"],
                relation_type=fields["relation_type"], value=fields["value"],
                datatype=fields["datatype"], source=fields["source"],
                source_memory_id=source_memory_id,
                evidence_memory_ids=(source_memory_id,),
                provenance=fields["provenance"], observed_at=fields["observed_at"],
                valid_from=fields["valid_from"], valid_to=fields["valid_to"],
                confidence=fields["confidence"],
            ))
        return prepared

    def _register_label(self, subject_id: str, label: Optional[str]) -> None:
        if label is None:
            return
        self.labels_by_id.setdefault(subject_id, set()).add(label)
        self.label_index.setdefault(label.casefold(), set()).add(subject_id)

    def commit(self, prepared: Iterable[SemanticFact]) -> None:
        """Add validated facts; merge repeated evidence without losing citations."""
        for fact in prepared:
            existing = self.facts.get(fact.id)
            if existing is not None:
                if (existing.subject_id, existing.relation_type, existing.value,
                    existing.datatype, existing.source, existing.provenance,
                    existing.observed_at, existing.valid_from, existing.valid_to,
                    existing.confidence) != (
                    fact.subject_id, fact.relation_type, fact.value, fact.datatype,
                    fact.source, fact.provenance, fact.observed_at, fact.valid_from,
                    fact.valid_to, fact.confidence,
                ):
                    raise ValueError("semantic fact ID collision")
                evidence = existing.evidence_memory_ids or (existing.source_memory_id,)
                if fact.source_memory_id not in evidence:
                    evidence = evidence + (fact.source_memory_id,)
                self.facts[fact.id] = replace(
                    existing, evidence_memory_ids=evidence,
                    subject_label=existing.subject_label or fact.subject_label,
                )
            else:
                self.facts[fact.id] = fact
            self.by_subject_relation.setdefault((fact.subject_id, fact.relation_type), set()).add(fact.id)
            self._register_label(fact.subject_id, fact.subject_label)

    def rebuild_indices(self, memories: Optional[Dict[str, Any]] = None) -> None:
        """Recover typed indexes from durable facts, fully revalidating each one.

        Every persisted fact is checked against the same canonical schema-v1
        rules (and ID derivation) as the write path before anything is
        published. ``memories``, when supplied, must contain every evidence ID
        and each linked memory's source must match the fact source.

        All work happens in local staging maps: a rejected rebuild raises
        ValueError and leaves the previously published indexes untouched.
        """
        if not isinstance(getattr(self, "facts", None), dict):
            raise ValueError("typed facts are missing or malformed; refusing data loss")
        old_labels = getattr(self, "labels_by_id", None)
        if not isinstance(old_labels, dict):
            raise ValueError("semantic labels are malformed; refusing data loss")
        if memories is not None and not isinstance(memories, dict):
            raise ValueError("memories must be a mapping of memory ID to memory")

        new_by_subject_relation: Dict[Tuple[str, str], set] = {}
        new_label_index: Dict[str, set] = {}
        new_labels_by_id: Dict[str, set] = {}
        active_subjects = set()

        def _stage_label(subject_id: str, label: Optional[str],
                         labels_by_id: Dict[str, set],
                         label_index: Dict[str, set]) -> None:
            if label is None:
                return
            labels_by_id.setdefault(subject_id, set()).add(label)
            label_index.setdefault(label.casefold(), set()).add(subject_id)

        for fact_id, fact in self.facts.items():
            if not isinstance(fact, SemanticFact):
                raise ValueError("malformed semantic fact; refusing data loss")
            if not isinstance(fact_id, str) or not fact_id:
                raise ValueError("malformed semantic fact key; refusing data loss")
            # A v1 pickle stores even optional fields explicitly as None.
            # getattr() can fall through to dataclass class defaults when an
            # instance attribute is deleted, concealing a truncated pickle.
            if set(vars(fact)) != _FACT_V1_ATTRS:
                raise ValueError("persisted semantic fact fields do not match schema v1")
            evidence = fact.evidence_memory_ids
            if type(evidence) is not tuple or not evidence:
                raise ValueError(
                    "stored evidence_memory_ids must be a nonempty tuple of memory IDs"
                )
            seen_evidence: List[str] = []
            for mem_id in evidence:
                if not isinstance(mem_id, str) or not mem_id.strip():
                    raise ValueError("stored evidence id must be nonblank text")
                if mem_id in seen_evidence:
                    raise ValueError("stored evidence ids must be unique")
                seen_evidence.append(mem_id)
            canonical = _validate_canonical_fact(
                subject_id=fact.subject_id,
                relation_type=fact.relation_type,
                value=fact.value,
                datatype=fact.datatype,
                source=fact.source,
                subject_label=fact.subject_label,
                provenance=fact.provenance,
                observed_at=fact.observed_at,
                valid_from=fact.valid_from,
                valid_to=fact.valid_to,
                confidence=fact.confidence,
                persisted=True,
            )
            source_memory_id = fact.source_memory_id
            if not isinstance(source_memory_id, str) or not source_memory_id.strip():
                raise ValueError("stored source_memory_id must be nonblank text")
            if source_memory_id not in seen_evidence:
                raise ValueError(
                    "stored source_memory_id must appear in evidence_memory_ids"
                )
            derived_id = _derive_fact_id(
                canonical["subject_id"], canonical["relation_type"],
                canonical["value"], canonical["datatype"], canonical["source"],
                canonical["provenance"], canonical["observed_at"],
                canonical["valid_from"], canonical["valid_to"],
                canonical["confidence"],
            )
            if fact_id != fact.id or fact_id != derived_id:
                raise ValueError("persisted semantic fact ID is not canonical; refusing data loss")
            if memories is not None:
                for mem_id in seen_evidence:
                    memory = memories.get(mem_id)
                    if memory is None:
                        raise ValueError("typed fact has missing source memory; refusing data loss")
                    if getattr(memory, "source", None) != canonical["source"]:
                        raise ValueError(
                            "typed fact source disagrees with its linked memory; "
                            "refusing data loss"
                        )
            active_subjects.add(canonical["subject_id"])
            new_by_subject_relation.setdefault(
                (canonical["subject_id"], canonical["relation_type"]), set()
            ).add(fact_id)
            _stage_label(canonical["subject_id"], canonical["subject_label"],
                         new_labels_by_id, new_label_index)

        # Preserve valid historical alias labels, but reject any orphan key,
        # unsupported container or noncanonical label instead of dropping it.
        # Pre-Layer-5 typed facts cannot be deleted, so an orphaned alias is
        # evidence of malformed persisted state, not routine stale cache data.
        for subject_id, old_subject_labels in old_labels.items():
            _entity_type(subject_id)
            if subject_id not in active_subjects:
                raise ValueError("stored semantic label has no active subject")
            if type(old_subject_labels) is not set:
                raise ValueError("stored semantic label cache is malformed; refusing data loss")
            for label in old_subject_labels:
                if not isinstance(label, str) or not label.strip() or label != label.strip():
                    raise ValueError("stored semantic label is malformed")
                _stage_label(subject_id, label, new_labels_by_id, new_label_index)

        # Publish only after every fact, link, and label passed validation.
        self.by_subject_relation = new_by_subject_relation
        self.label_index = new_label_index
        self.labels_by_id = new_labels_by_id

    def query(self, subject_id: str, relation_type: str,
              observed_as_of: Any = None, valid_at: Any = None) -> List[SemanticFact]:
        _validate_subject_relation(subject_id, relation_type)
        observed_cutoff = _utc(observed_as_of, "observed_as_of")
        valid_cutoff = _utc(valid_at, "valid_at")
        matches = []
        for fact_id in self.by_subject_relation.get((subject_id, relation_type), ()):
            fact = self.facts[fact_id]
            if observed_cutoff is not None and fact.observed_at > observed_cutoff:
                continue
            if valid_cutoff is not None:
                if fact.valid_from is not None and fact.valid_from > valid_cutoff:
                    continue
                if fact.valid_to is not None and valid_cutoff >= fact.valid_to:
                    continue
            matches.append(fact)
        # No winner: return *every* matching assertion, in stable display order.
        return sorted(matches, key=lambda fact: (fact.observed_at, fact.id), reverse=True)

    def resolve_entity(self, label: str, entity_type: Optional[str] = None) -> str:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("entity label must be nonblank text")
        if entity_type is not None and entity_type not in _ENTITY_TYPES:
            raise ValueError("entity_type is not allowlisted")
        matches = self.label_index.get(label.strip().casefold(), set())
        if entity_type is not None:
            matches = {subject_id for subject_id in matches
                       if _entity_type(subject_id) == entity_type}
        if not matches:
            raise LookupError("unknown entity label; provide an explicit subject_id")
        if len(matches) > 1:
            raise AmbiguousEntityError("display label is ambiguous; provide an explicit subject_id")
        return next(iter(matches))
