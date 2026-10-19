"""DK-canary HTTP admission and private exact-state digest helpers.

The gate covers HTTP mutations only. It is not a substitute for freezing all
external DK writers before an operator captures a production graph pair.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from typing import Any

import numpy as np


def canonical_bytes(value: Any) -> bytes:
    """One deterministic, strict UTF-8 encoding for inventories and receipts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _json_scalar(value: Any) -> Any:
    """Preserve complete graph edge attributes without lossy repr/default=str."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite edge attribute")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("non-string edge attribute key")
        return {key: _json_scalar(item) for key, item in value.items()}
    raise ValueError("unsupported edge attribute type")


def graph_inventory_v1(graph) -> dict:
    """Historical SUBSET digest: ID/source/text, edges and typed facts only.

    This is never an exact-state receipt. New maintenance paths must use the
    versioned graph_inventory() below; retained solely for v1 diagnostics.
    """
    memories = []
    for mid, memory in sorted(graph.memories.items()):
        if (not isinstance(mid, str) or not isinstance(memory.source, str)
                or not isinstance(memory.text, str) or memory.id != mid):
            raise ValueError("malformed live memory inventory")
        memories.append([mid, memory.source, memory.text])
    if set(graph.graph.nodes) != set(graph.memories):
        raise ValueError("graph nodes disagree with memory inventory")
    edges = []
    for left, right, attributes in graph.graph.edges(data=True):
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError("non-string graph node ID")
        low, high = sorted((left, right))
        edges.append([low, high, _json_scalar(attributes)])
    edges.sort(key=lambda row: (row[0], row[1]))
    facts = []
    for fid, fact in sorted(graph.semantic_store.facts.items()):
        if not isinstance(fid, str) or fact.id != fid:
            raise ValueError("malformed typed fact ID")
        facts.append([fid, fact.to_dict()])
    full = {"memories": memories, "edges": edges, "typed_facts": facts}
    return {
        "counts": {
            "memories": len(memories),
            "nodes": graph.graph.number_of_nodes(),
            "edges": len(edges),
            "typed_facts": len(facts),
        },
        "sha256": {
            "memories": sha256(memories),
            "edges": sha256(edges),
            "typed_facts": sha256(facts),
            "full": sha256(full),
        },
    }


def graph_inventory(graph) -> dict:
    """Only v2 is eligible to attest exact loaded-state parity."""
    from .inventory_v2 import graph_inventory_v2
    return graph_inventory_v2(graph)


class AdmissionGate:
    """Atomic ingress close + bounded wait across all gated HTTP requests.

    The condition protects admission, active-count, denial audit and control
    transitions. Never hold it over a handler, disk I/O, or an async await.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self.paused = os.environ.get("QMG_SEMANTIC_START_PAUSED") == "1"
        self.inflight = 0
        self.denied_writes = 0
        self.receipt: dict | None = None
        self.working = False

    def reset(self) -> None:
        """Test-only isolation; never invoked by an HTTP endpoint."""
        with self._condition:
            if self.inflight or self.working:
                raise RuntimeError("cannot reset while admissions are active")
            self.paused = os.environ.get("QMG_SEMANTIC_START_PAUSED") == "1"
            self.denied_writes = 0
            self.receipt = None

    def arm_start_paused(self) -> None:
        """Startup only closes; it can never implicitly reopen a paused gate."""
        if os.environ.get("QMG_SEMANTIC_START_PAUSED") == "1":
            with self._condition:
                self.paused = True
                self.receipt = None

    def is_paused(self) -> bool:
        with self._condition:
            return self.paused

    def enter(self) -> bool:
        with self._condition:
            if self.paused:
                self.denied_writes += 1
                if self.receipt is not None:
                    updated = {key: value for key, value in self.receipt.items()
                               if key != "receipt_sha256"}
                    updated["denied_writes"] = self.denied_writes
                    self.receipt = {**updated, "receipt_sha256": sha256(updated)}
                return False
            self.inflight += 1
            return True

    def leave(self) -> None:
        with self._condition:
            if self.inflight <= 0:
                raise RuntimeError("admission accounting underflow")
            self.inflight -= 1
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self.working:
                raise RuntimeError("another maintenance transition is active")
            if not self.paused:
                self.denied_writes = 0
            self.paused = True
            self.receipt = None
            self.working = True
            self._condition.notify_all()

    def wait_drained(self, timeout: float) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: self.inflight == 0,
                                            timeout=timeout)

    def finish_pause(self, receipt: dict | None) -> None:
        with self._condition:
            if receipt is not None:
                body = {key: value for key, value in receipt.items()
                        if key != "receipt_sha256"}
                body["denied_writes"] = self.denied_writes
                receipt = {**body, "receipt_sha256": sha256(body)}
            self.receipt = receipt
            self.working = False
            self._condition.notify_all()

    def status(self) -> dict:
        with self._condition:
            return {
                "paused": self.paused,
                "inflight": self.inflight,
                "denied_writes": self.denied_writes,
                "receipt_verified": bool(self.receipt and not self.working),
            }

    def current_receipt(self) -> dict | None:
        with self._condition:
            return self.receipt if self.paused and not self.working else None

    def invalidate_receipt(self) -> None:
        with self._condition:
            self.receipt = None

    def start_resume(self, receipt_sha256: str) -> dict:
        with self._condition:
            if (not self.paused or self.working or self.inflight
                    or self.receipt is None
                    or self.receipt["receipt_sha256"] != receipt_sha256):
                raise RuntimeError("no current verified receipt for resume")
            if self.denied_writes:
                raise RuntimeError("denied writes require an upstream audit; NO-GO")
            self.working = True
            return self.receipt

    def finish_resume(self, verified: bool, expected_sha256: str) -> bool:
        with self._condition:
            # A rejected writer arriving DURING disk verification is a NO-GO,
            # even if no rejection existed when the operator requested resume.
            safe = (verified and self.paused and not self.inflight
                    and self.denied_writes == 0 and self.receipt is not None
                    and self.receipt["receipt_sha256"] == expected_sha256)
            if safe:
                self.paused = False
                self.receipt = None
                self.denied_writes = 0
            elif verified:
                # Keep the still-valid snapshot visible for denied-write audit;
                # reopening requires a new pause cycle after writer freeze.
                pass
            else:
                self.receipt = None
            self.working = False
            self._condition.notify_all()
            return safe
