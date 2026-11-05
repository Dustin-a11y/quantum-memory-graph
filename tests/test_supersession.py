"""Supersession: derived state over memories that accumulated instead of replacing.

These tests pin the behaviour that matters and the behaviour that would be dangerous:
a claim is only superseded by a genuinely different value for the same (subject, attribute), a
restatement never supersedes, history questions resolve to the replaced value, pointer claims are
followed, and a store with nothing to supersede is left completely alone.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantum_memory_graph.graph import MemoryGraph
from quantum_memory_graph.supersession import (StateIndex, parse_claim,
                                               state_aware_search, _norm)

T0 = datetime(2023, 1, 1, 9, 0, 0)


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


def build(rows):
    graph = MemoryGraph()
    graph._embedder = TinyEmbedder()
    for text, when in rows:
        mem = graph.add_memory(text, timestamp=when, source="test")
        mem.embedding = graph._embedder.encode([text])[0]
    return graph


def test_articles_and_case_do_not_split_an_entity():
    assert _norm("The Rack") == _norm("the rack") == _norm("the the rack") == "rack"


@pytest.mark.parametrize("text,subject,attribute,value", [
    ("the alpha server is located in Seattle.", "alpha server", "located", "Seattle"),
    ("Dustin's favorite editor is Neovim.", "dustin", "favorite editor", "Neovim"),
    ("Dustin lives in Seattle.", "dustin", "lives in", "Seattle"),
    ("the epsilon replica 0's primary is epsilon main 0.", "epsilon replica 0", "primary",
     "epsilon main 0"),
])
def test_parse_claim_shapes(text, subject, attribute, value):
    claim = parse_claim(text, "id", T0)
    assert claim is not None
    assert (claim.subject, claim.attribute, claim.value) == (subject, attribute, value)


@pytest.mark.parametrize("text", [
    "",
    "ok",
    "The cron job status was completed.",
    "GraphQL is a query language.",
    "x" * 500,
])
def test_parser_refuses_non_state_sentences(text):
    assert parse_claim(text, "id", T0) is None


def test_latest_value_wins_and_older_is_superseded():
    index = StateIndex.from_graph(build([
        ("the alpha server is located in Seattle.", T0),
        ("the alpha server is located in Portland.", T0 + timedelta(days=30)),
    ]))
    current = index.current("the alpha server", "located")
    assert current is not None and current.value == "Portland"
    history = index.history("alpha server", "located")
    assert [c.value for c in history] == ["Seattle", "Portland"]
    assert history[0].superseded_by == history[1].memory_id
    assert history[0].valid_to == history[1].valid_from
    assert history[1].is_current and not history[0].is_current


def test_restatement_does_not_supersede():
    index = StateIndex.from_graph(build([
        ("the theta unit 0 is located in Austin.", T0),
        ("the theta unit 0 is located in Austin.", T0 + timedelta(days=9)),
    ]))
    assert all(c.is_current for c in index.history("theta unit 0", "located"))


def test_current_question_resolves_to_the_live_value():
    graph = build([
        ("the alpha server is located in Seattle.", T0),
        ("the alpha server is located in Portland.", T0 + timedelta(days=30)),
    ])
    index = StateIndex.from_graph(graph)
    resolved = index.resolve("Where is the alpha server located now?")
    assert resolved["matched"] and resolved["intent"] == "current"
    assert resolved["current"]["value"] == "Portland"
    top = state_aware_search("Where is the alpha server located now?", graph, k=3, index=index)[0]
    assert "Portland" in top["text"]


def test_history_question_resolves_to_the_replaced_value():
    graph = build([
        ("the delta node is located in Denver.", T0),
        ("the delta node is located in Boston.", T0 + timedelta(days=20)),
        ("the delta node is located in Miami.", T0 + timedelta(days=40)),
    ])
    index = StateIndex.from_graph(graph)
    resolved = index.resolve("Where was the delta node located before that?")
    assert resolved["intent"] == "history"
    assert resolved["previous"]["value"] == "Boston"
    top = state_aware_search("Where was the delta node located before that?", graph, k=3, index=index)[0]
    assert "Boston" in top["text"]


def test_pointer_claims_are_followed():
    graph = build([
        ("the epsilon main 0 is located in Denver.", T0),
        ("the epsilon replica 0's primary is epsilon main 0.", T0 + timedelta(days=10)),
        ("the epsilon main 0 is located in Miami.", T0 + timedelta(days=40)),
    ])
    index = StateIndex.from_graph(graph)
    resolved = index.resolve("Where is the epsilon replica 0 located now?")
    assert resolved["matched"] and resolved["hops"] >= 1
    assert resolved["current"]["value"] == "Miami"


def test_no_state_change_means_no_interference():
    graph = build([("the zeta unit 0 is located in Seattle.", T0)])
    index = StateIndex.from_graph(graph)
    top = state_aware_search("Where is the zeta unit 0 located now?", graph, k=3, index=index)[0]
    assert top.get("source") != "supersession-resolution"
