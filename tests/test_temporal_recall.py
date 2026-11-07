"""Temporal recall: timestamps survive a store, and search can filter/rank by time.

These tests pin the two guarantees that make time-aware recall safe:

* a stored memory always carries a timestamp (naive UTC), so nothing downstream has to
  special-case ``None``;
* asking for ``recency_weight=0.0`` is *exactly* today's ranking — same ids, same scores.
  A recency feature that drifts the default ordering is a silent recall regression.

The rest pins the contract of ``time_range`` (inclusive bounds, one-sided bounds, no-timestamp
memories excluded) and of the per-result ``timestamp`` / ``age_days`` fields.
"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from quantum_memory_graph import pipeline
from quantum_memory_graph.graph import MemoryGraph
from quantum_memory_graph.search import hybrid_search

UTC = timezone.utc


class TinyEmbedder:
    """Deterministic bag-of-words encoder so tests never download a model."""

    dims = 128

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in (text or "").lower().split():
                out[i, (hash(token) % self.dims)] += 1.0
            norm = np.linalg.norm(out[i])
            if normalize_embeddings and norm:
                out[i] /= norm
        return out


def new_graph():
    """A MemoryGraph with the tiny encoder installed (no sentence-transformer)."""
    graph = MemoryGraph()
    graph._embedder = TinyEmbedder()
    return graph


def seed(graph, rows):
    """rows: (text, timestamp) — timestamp may be None to model a legacy memory."""
    for text, when in rows:
        mem = graph.add_memory(text, timestamp=when, source="test")
        mem.timestamp = when  # keep None as None: a genuinely undated memory
        mem.embedding = graph._embedder.encode([text])[0]
    return graph


def utc_now_naive():
    return datetime.utcnow()


# --------------------------------------------------------------------------- store


class TestStoreDefaultsTimestamp:
    def test_store_without_timestamp_gets_utc_now(self):
        graph = new_graph()
        before = utc_now_naive()
        result = pipeline.store("the beta router came online.", graph=graph)
        after = utc_now_naive()

        mem = graph.memories[result["memory_id"]]
        assert mem.timestamp is not None
        assert mem.timestamp.tzinfo is None, "stored timestamps must be naive UTC"
        assert before - timedelta(seconds=5) <= mem.timestamp <= after + timedelta(seconds=5)

    def test_store_never_leaves_timestamp_none(self):
        graph = new_graph()
        result = pipeline.store("the gamma cache was flushed.", graph=graph)
        assert graph.memories[result["memory_id"]].timestamp is not None

    def test_explicit_timestamp_is_honoured(self):
        graph = new_graph()
        when = datetime(2025, 5, 4, 7, 30, 0)
        result = pipeline.store("the delta worker restarted.", timestamp=when, graph=graph)
        assert graph.memories[result["memory_id"]].timestamp == when

    def test_tz_aware_timestamp_becomes_naive_utc(self):
        graph = new_graph()
        when = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=-8)))
        result = pipeline.store("the epsilon queue drained.", timestamp=when, graph=graph)
        stored = graph.memories[result["memory_id"]].timestamp
        assert stored.tzinfo is None
        assert stored == datetime(2025, 6, 1, 20, 0, 0)

    def test_utc_aware_timestamp_is_not_shifted(self):
        graph = new_graph()
        when = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        result = pipeline.store("the zeta index rebuilt.", timestamp=when, graph=graph)
        assert graph.memories[result["memory_id"]].timestamp == datetime(2025, 6, 1, 12, 0, 0)


class TestStoreBatchTimestamps:
    def test_batch_defaults_every_timestamp(self):
        graph = new_graph()
        before = utc_now_naive()
        result = pipeline.store_batch([
            "the eta node joined the cluster.",
            "the theta node left the cluster.",
            "the iota node was replaced.",
        ], graph=graph)
        after = utc_now_naive()

        stored = [graph.memories[mid] for mid in result["memory_ids"]]
        assert len(stored) == 3
        for mem in stored:
            assert mem.timestamp is not None
            assert mem.timestamp.tzinfo is None
            assert before - timedelta(seconds=5) <= mem.timestamp <= after + timedelta(seconds=5)

    def test_batch_honours_positional_timestamps(self):
        graph = new_graph()
        stamps = [
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 2, 2, 0, 0, 0),
            datetime(2024, 3, 3, 0, 0, 0),
        ]
        result = pipeline.store_batch([
            "the kappa job was scheduled.",
            "the lambda job was scheduled.",
            "the mu job was scheduled.",
        ], timestamps=stamps, graph=graph)

        got = [graph.memories[mid].timestamp for mid in result["memory_ids"]]
        assert got == stamps


# --------------------------------------------------------------------------- save_json


class TestSaveJsonCarriesTimestamp:
    def test_entries_include_timestamp_and_keep_existing_keys(self, tmp_path):
        graph = new_graph()
        when = datetime(2025, 2, 3, 4, 5, 6)
        mem = graph.add_memory("the nu server moved racks.", timestamp=when, source="ops")
        path = tmp_path / "graph.json"
        graph.save_json(str(path))

        data = json.loads(path.read_text())
        assert data["memories"], "save_json wrote no memories"
        entry = next(e for e in data["memories"] if e["id"] == mem.id)
        assert set(entry) == {"id", "text", "source", "timestamp"}
        assert entry["id"] == mem.id
        assert entry["text"] == "the nu server moved racks."
        assert entry["source"] == "ops"
        assert entry["timestamp"] == when.isoformat()

    def test_timestamp_round_trips_through_json(self, tmp_path):
        graph = new_graph()
        when = datetime(2025, 2, 3, 4, 5, 6)
        graph.add_memory("the xi server moved racks.", timestamp=when, source="ops")
        path = tmp_path / "graph.json"
        graph.save_json(str(path))

        data = json.loads(path.read_text())
        parsed = datetime.fromisoformat(data["memories"][0]["timestamp"])
        assert parsed == when

    def test_undated_memory_serialises_as_null(self, tmp_path):
        graph = new_graph()
        mem = graph.add_memory("the omicron note has no date.", source="ops")
        mem.timestamp = None  # a legacy memory that genuinely has none
        path = tmp_path / "graph.json"
        graph.save_json(str(path))

        data = json.loads(path.read_text())
        entry = next(e for e in data["memories"] if e["id"] == mem.id)
        assert "timestamp" in entry
        assert entry["timestamp"] is None


# --------------------------------------------------------------------------- time_range


MARCH = [
    ("the pi cluster incident opened.", datetime(2025, 3, 1, 0, 0, 0)),
    ("the pi cluster incident escalated.", datetime(2025, 3, 10, 0, 0, 0)),
    ("the pi cluster incident resolved.", datetime(2025, 3, 20, 0, 0, 0)),
    ("the inconsistency is alarming.", datetime(2025, 4, 15, 0, 0, 0)),
]
ALL_TS = [ts for _, ts in MARCH]


class TestTimeRangeFilter:
    @classmethod
    def setup_class(cls):
        cls.graph = seed(new_graph(), MARCH)
        cls.graph.add_memory("the rho note predates tracking.", source="test")
        cls.graph.memories[
            MemoryGraphId(cls.graph, "the rho note predates tracking.")
        ].timestamp = None
        cls.undated_id = MemoryGraphId(cls.graph, "the rho note predates tracking.")
        cls.all_ids = {mid for mid in cls.graph.memories}

    def test_bounds_are_inclusive(self):
        got = hybrid_search(
            "pi cluster incident", self.graph, k=10,
            time_range={"from": ALL_TS[0].isoformat(), "to": ALL_TS[1].isoformat()},
        )
        ids = {r["text"] for r in got}
        # both boundary memories must survive the filter
        assert MARCH[0][0] in ids
        assert MARCH[1][0] in ids
        assert MARCH[2][0] not in ids

    def test_from_only(self):
        got = hybrid_search(
            "pi cluster incident", self.graph, k=10,
            time_range={"from": ALL_TS[1].isoformat()},
        )
        ids = {r["text"] for r in got}
        assert MARCH[0][0] not in ids
        assert MARCH[1][0] in ids and MARCH[2][0] in ids

    def test_to_only(self):
        got = hybrid_search(
            "pi cluster incident", self.graph, k=10,
            time_range={"to": ALL_TS[1].isoformat()},
        )
        ids = {r["text"] for r in got}
        assert MARCH[0][0] in ids and MARCH[1][0] in ids
        assert MARCH[2][0] not in ids

    def test_undated_memory_is_excluded_when_filtering(self):
        got = hybrid_search(
            "note predates tracking", self.graph, k=10,
            time_range={"from": ALL_TS[0].isoformat()},
        )
        assert all(r["id"] != self.undated_id for r in got)
        # ... and is still reachable when no filter is asked for
        near = hybrid_search("note predates tracking", self.graph, k=10)
        assert any(r["id"] == self.undated_id for r in near)

    @pytest.mark.parametrize("bad", ["not-a-date", "2025-13-40T00:00:00", ""])
    def test_bad_from_iso_raises(self, bad):
        with pytest.raises(ValueError):
            hybrid_search("pi cluster incident", self.graph, k=5,
                          time_range={"from": bad})

    @pytest.mark.parametrize("bad", ["yesterday", "2025-02-30T00:00:00"])
    def test_bad_to_iso_raises(self, bad):
        with pytest.raises(ValueError):
            hybrid_search("pi cluster incident", self.graph, k=5,
                          time_range={"to": bad})


def MemoryGraphId(graph, text):
    """The id add_memory would assign; used to mutate a seeded memory in place."""
    from quantum_memory_graph.graph import Memory
    for mid, mem in graph.memories.items():
        if mem.text == text:
            return mid
    return Memory.make_id(text)


# --------------------------------------------------------------------------- recency


# Relative to now on purpose: the recency bonus decays with exp(-age_days/30), so fixed 2025 dates
# would give every memory a decay of ~0 and the test would assert nothing about recency at all.
_NOW = datetime.utcnow()
RECENT = [
    ("the sigma rollout step one reported.", _NOW - timedelta(days=300)),
    ("the sigma rollout step two reported.", _NOW - timedelta(days=240)),
    ("the sigma rollout step three reported.", _NOW - timedelta(days=180)),
    ("the sigma rollout step four reported.", _NOW - timedelta(days=120)),
    ("the sigma rollout step five reported.", _NOW - timedelta(days=60)),
    ("the sigma rollout step six reported.", _NOW - timedelta(days=1)),
]


class TestRecencyWeight:
    @classmethod
    def setup_class(cls):
        cls.graph = seed(new_graph(), RECENT)
        cls.newest_id = cls.graph.memories[MemoryGraphId(cls.graph, RECENT[-1][0])].id
        cls.oldest_id = cls.graph.memories[MemoryGraphId(cls.graph, RECENT[0][0])].id

    def test_zero_recency_is_identical_to_legacy_call(self):
        legacy = hybrid_search("sigma rollout step reported", self.graph, k=6)
        explicit = hybrid_search("sigma rollout step reported", self.graph, k=6,
                                 recency_weight=0.0)
        assert [r["id"] for r in explicit] == [r["id"] for r in legacy]
        assert [r["combined_score"] for r in explicit] == \
               [r["combined_score"] for r in legacy]

    def test_none_recency_is_identical_to_legacy_call(self):
        legacy = hybrid_search("sigma rollout step reported", self.graph, k=6)
        explicit = hybrid_search("sigma rollout step reported", self.graph, k=6,
                                 recency_weight=0.0, time_range=None)
        assert [r["id"] for r in explicit] == [r["id"] for r in legacy]
        assert [r["combined_score"] for r in explicit] == \
               [r["combined_score"] for r in legacy]

    def test_positive_recency_prefers_newer(self):
        legacy = hybrid_search("sigma rollout step reported", self.graph, k=6)
        boosted = hybrid_search("sigma rollout step reported", self.graph, k=6,
                               recency_weight=0.5)
        assert boosted and boosted[0]["id"] == self.newest_id
        assert boosted[-1]["id"] == self.oldest_id
        # the default ranking is untuned, so say nothing about legacy's first id
        assert len(legacy) == len(boosted)

    def test_bonus_is_bounded_by_recency_weight(self):
        legacy = {r["id"]: r["combined_score"]
                  for r in hybrid_search("sigma rollout step reported", self.graph, k=6)}
        boosted = hybrid_search("sigma rollout step reported", self.graph, k=6,
                               recency_weight=0.25)
        for r in boosted:
            base = legacy[r["id"]]
            # multiplicative: the bonus is proportional, so it is bounded by (1 + weight) of the
            # base score rather than by an absolute amount — the base is not on a fixed scale.
            assert r["combined_score"] >= base - 1e-9
            assert r["combined_score"] <= base * (1.0 + 0.25) + 1e-6

    def test_results_carry_timestamp_and_age_days(self):
        boosted = hybrid_search("sigma rollout step reported", self.graph, k=6,
                                recency_weight=0.1)
        assert boosted
        for r in boosted:
            assert "timestamp" in r and "age_days" in r
            assert isinstance(r["timestamp"], str)
            assert isinstance(r["age_days"], float)
        # the oldest memory is ~= however many days before now
        oldest = next(r for r in boosted if r["id"] == self.oldest_id)
        expected = (datetime.utcnow() - RECENT[0][1]).total_seconds() / 86400.0
        assert abs(oldest["age_days"] - expected) < 1.0

    def test_undated_result_reports_null_age(self):
        graph = seed(new_graph(), RECENT[:2])
        undated = graph.add_memory("the tau footnote has no date.", source="test")
        undated.timestamp = None
        got = hybrid_search("footnote has no date", graph, k=5, recency_weight=0.0)
        assert got
        row = next((r for r in got if r["id"] == undated.id), None)
        assert row is not None, "undated memory dropped from an unfiltered search"
        assert row["timestamp"] is None
        assert row["age_days"] is None
