"""API surface + error-hygiene regressions.

Two properties are pinned here:

1. The dead source-node routes are gone. ``MemoryGraph`` has no
   ``add_source``/``sources``/``list_sources``/``get_source_memories`` in any
   generation of the package, so every call to those routes raised
   ``AttributeError``. They must stay absent (404), not come back as stubs.

2. Validation failures must not echo internal exception text to the caller.
   The exception is logged server-side and the response carries a generic,
   still-useful reason.

No embedding model is loaded and no network is used: the app is imported and
exercised through ``TestClient`` without its lifespan, and the semantic
validation seam is monkeypatched so the 422 branch is reached deterministically.
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quantum_memory_graph import api as api_module
from quantum_memory_graph.api import app

API_FILE = Path(api_module.__file__)

CANARY_TOKEN = "test-canary-token"
SENTINEL = "INTERNAL-ONLY-DETAIL-9f3a1c"


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def canary(monkeypatch):
    """Enable the DK semantic canary for one test without touching real env."""
    monkeypatch.setenv("QMG_API_TOKEN", CANARY_TOKEN)
    monkeypatch.setenv("QMG_SEMANTIC_CANARY", "dk")
    monkeypatch.setattr(api_module, "API_TOKEN", CANARY_TOKEN, raising=False)
    return {"Authorization": f"Bearer {CANARY_TOKEN}"}


# ── 1. removed surface ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/source"),
        ("get", "/sources"),
        ("get", "/source/abc/graph"),
    ],
)
def test_removed_source_routes_are_not_registered(client, method, path):
    response = getattr(client, method)(path)
    assert response.status_code == 404, response.text


def test_router_has_no_source_paths():
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/source" not in paths
    assert "/sources" not in paths
    assert "/source/{source_id}/graph" not in paths
    assert not [p for p in paths if p.startswith("/source")]


def test_dead_source_symbols_are_absent_from_module():
    source = API_FILE.read_text(encoding="utf-8")
    # `\b` keeps the unrelated DeleteBySourceRequest model out of this check.
    assert re.search(r"\bSourceRequest\b", source) is None
    assert "add_source" not in source
    assert "api_add_source" not in source
    assert "api_list_sources" not in source
    assert "api_source_graph" not in source
    assert '_semantic' in source  # module really did load the file we scanned


# ── 2. exception text is not exposed to callers ─────────────────────


def test_semantic_recall_422_hides_exception_text(client, canary, monkeypatch):
    def boom(subject_id, relation_type):
        raise ValueError(f"{SENTINEL}: relation_type is not allowlisted")

    monkeypatch.setattr(api_module._semantic, "_validate_subject_relation", boom)

    response = client.post(
        "/semantic/recall",
        json={"subject_id": "person:alice", "relation_type": "nope"},
        headers=canary,
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert detail == "invalid subject_id or relation_type"
    assert SENTINEL not in detail
    assert SENTINEL not in response.text
    assert "Traceback" not in response.text
    assert 'File "' not in response.text
    assert "ValueError" not in detail


def test_semantic_store_422_hides_exception_text(client, canary, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError(f"{SENTINEL}: datatype is not permitted for relation_type")

    monkeypatch.setattr(api_module._semantic.SemanticStore, "prepare", boom)

    response = client.post(
        "/semantic/store",
        json={
            "text": "alice works at acme",
            "source": "test",
            "semantic_facts": [
                {
                    "subject_id": "person:alice",
                    "relation_type": "employed_by",
                    "datatype": "string",
                    "value": "acme",
                }
            ],
        },
        headers=canary,
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert detail == "invalid semantic fact payload"
    assert SENTINEL not in response.text
    assert "Traceback" not in response.text
    assert 'File "' not in response.text


def test_timestamp_422_hides_exception_text(client, canary):
    """A real (unmonkeypatched) validation failure also stays generic."""
    response = client.post(
        "/semantic/recall",
        json={
            "subject_id": "person:alice",
            "relation_type": "employed_by",
            "observed_as_of": "not-a-timestamp",
        },
        headers=canary,
    )
    if response.status_code == 404:
        pytest.skip("semantic canary not reachable in this environment")
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "not-a-timestamp" not in detail
    assert "Traceback" not in response.text


def test_no_exception_interpolation_in_http_error_details():
    """Source guard: no route builds an HTTPException detail from an exception."""
    pattern = re.compile(r"detail\s*=\s*(str\(|f[\"']).*\b(e|exc|err)\b")
    offenders = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(API_FILE.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == [], offenders
    assert "detail=str(" not in API_FILE.read_text(encoding="utf-8")


# ── 3. remaining exception-text sinks ──────────────────────────────


def test_learn_error_path_hides_exception_text(client, monkeypatch):
    class Boom:
        def learn(self, min_co_recalls=1):
            raise RuntimeError(f"{SENTINEL}: learn exploded")

    monkeypatch.setattr(api_module, "_graph", Boom())

    response = client.post("/learn")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "learning failed"
    assert SENTINEL not in response.text
    assert "RuntimeError" not in response.text


def test_scores_error_path_hides_exception_text(client, monkeypatch):
    class Boom:
        def get_memory_scores(self):
            raise RuntimeError(f"{SENTINEL}: scores exploded")

    monkeypatch.setattr(api_module, "_graph", Boom())

    response = client.get("/scores")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "scores unavailable"
    assert SENTINEL not in response.text
    assert "RuntimeError" not in response.text


def _fake_graph():
    """A minimal graph stub: two duplicate memories, no embedding model."""
    import numpy as np
    from types import SimpleNamespace

    class FakeGraph:
        def __init__(self):
            self.memories = {
                "m1": SimpleNamespace(embedding=np.array([1.0, 0.0])),
                "m2": SimpleNamespace(embedding=np.array([1.0, 0.0])),
            }

        def merge_clusters(self, clusters, dry_run=False):
            return {"ok": True, "clusters_merged": len(clusters), "memories_merged": 2}

        def delete_by_source(self, sources, dry_run=False):
            return {"ok": True, "matched": 1, "removed": 1, "dry_run": dry_run}

    return FakeGraph()


def test_merge_duplicates_persist_failure_hides_exception_text(client, monkeypatch):
    monkeypatch.setattr(api_module, "get_graph", _fake_graph)

    def boom():
        raise RuntimeError(f"{SENTINEL}: disk full")

    monkeypatch.setattr(api_module, "_persist_current_graph", boom)

    response = client.post("/merge-duplicates", json={"threshold": 0.9})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["persisted"] is False
    assert body["persist_error"] == "persist failed; see server logs"
    assert SENTINEL not in response.text
    assert "RuntimeError" not in response.text


def test_delete_by_source_persist_failure_hides_exception_text(client, monkeypatch):
    monkeypatch.setattr(api_module, "get_graph", _fake_graph)

    def boom():
        raise RuntimeError(f"{SENTINEL}: disk full")

    monkeypatch.setattr(api_module, "_persist_current_graph", boom)

    response = client.post(
        "/delete-by-source",
        json={"sources": ["test-source"], "dry_run": False},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["persisted"] is False
    assert body["persist_error"] == "persist failed; see server logs"
    assert SENTINEL not in response.text
    assert "RuntimeError" not in response.text


def test_answer_llm_failure_does_not_expose_exception_text(monkeypatch):
    import json

    from quantum_memory_graph import pipeline as pipeline_module

    def fake_recall(query, K=5, graph=None):
        return {"memories": [{"text": "memory one", "relevance": 0.9}]}

    def boom(*args, **kwargs):
        raise RuntimeError(f"{SENTINEL}: llm exploded")

    monkeypatch.setattr(pipeline_module, "recall", fake_recall)
    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    result = pipeline_module.answer("what is x?", K=1)

    assert result["method"] == "llm_failed"
    assert SENTINEL not in json.dumps(result)
    assert "RuntimeError" not in json.dumps(result)
