"""Regression coverage for the staged benchmark's CodeQL dead-store finding."""

import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "benchmarks" / "run_longmemeval_staged.py"


def test_t0_assignments_are_used_before_reassignment():
    """Every t0 assignment must be read before another assignment replaces it."""
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    events = sorted(
        (
            node.lineno,
            "store" if isinstance(node.ctx, ast.Store) else "load",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "t0"
    )

    pending_store = None
    for line, kind in events:
        if kind == "store":
            assert pending_store is None, (
                f"t0 assigned at line {line} before the assignment at "
                f"line {pending_store} was read"
            )
            pending_store = line
        else:
            pending_store = None
