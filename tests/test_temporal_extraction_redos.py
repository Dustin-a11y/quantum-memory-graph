"""Regression: temporal extraction must stay linear on adversarial input.

Unbounded ``\\w+`` / ``\\d+`` quantifiers followed by a required ``\\s+`` backtrack
quadratically on long single-token text. A 50,000-digit memory string blocked
``_extract_temporal_entities`` for ~115s, which is a denial of service on any
service that accepts memory text (POST /memory calls it via add_memory).
The quantifiers are now length-capped, and these tests hold the line.
"""
import time

import pytest

from quantum_memory_graph.graph import MemoryGraph


@pytest.fixture(scope="module")
def graph():
    # Bypass __init__ so no embedding model is constructed.
    return MemoryGraph.__new__(MemoryGraph)


ADVERSARIAL = [
    ("digits", "9" * 50_000),
    ("words", "a " * 50_000),
    ("commas", "1," * 25_000),
]


@pytest.mark.parametrize("label,text", ADVERSARIAL)
def test_adversarial_text_extracts_quickly(graph, label, text):
    started = time.perf_counter()
    graph._extract_temporal_entities(text)
    elapsed = time.perf_counter() - started
    # Uncapped quantifiers took ~115s here; 2s leaves ample headroom on slow CI
    # while still failing loudly if an unbounded quantifier comes back.
    assert elapsed < 2.0, f"{label} extraction took {elapsed:.2f}s (quadratic backtracking is back)"


def test_scaling_is_not_quadratic(graph):
    """4x the input must not cost anywhere near 16x the time."""
    small = "9" * 10_000
    started = time.perf_counter()
    graph._extract_temporal_entities(small)
    small_elapsed = time.perf_counter() - started

    big = "9" * 40_000
    started = time.perf_counter()
    graph._extract_temporal_entities(big)
    big_elapsed = time.perf_counter() - started

    assert big_elapsed < max(small_elapsed * 8, 0.5), (
        f"4x input cost {big_elapsed / max(small_elapsed, 1e-9):.1f}x time (quadratic)")


def test_normal_extraction_unchanged(graph):
    assert graph._extract_temporal_entities("Rachel is 12 years old and waited 30 minutes") == [
        {"type": "age", "subject": "Rachel", "value": 12, "text": "Rachel is 12 years old"},
        {"type": "duration", "value": 12, "unit": "year", "text": "12 years"},
        {"type": "duration", "value": 30, "unit": "minute", "text": "30 minutes"},
    ]


def test_quantities_and_dates_still_found(graph):
    found = graph._extract_temporal_entities("Deadline 2026-08-15, order 412 pages for $15,000")
    values = {(item["type"], str(item.get("value"))) for item in found}
    assert ("date", "2026-08-15") in values
    assert ("quantity", "412.0") in values
