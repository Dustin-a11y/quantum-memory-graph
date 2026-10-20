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
