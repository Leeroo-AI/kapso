"""Pin the selector hardening: quality floor, retry, and K lanes never shrink.

The parser is loaded from THIS repo's ideation.py by AST extraction rather
than by import: another kapso checkout can shadow the package on this box,
and the pin must cover the file that actually ships here. Only the constant
and the pure parser are executed — they depend on `re` and `logger` alone.
"""

import ast
import logging
import re
from pathlib import Path

_IDEATION = (
    Path(__file__).resolve().parents[1]
    / "src/kapso/execution/search_strategies/generic/ideation.py"
)
_SOURCE = _IDEATION.read_text()

_WANTED = {"MIN_SELECTED_SOLUTION_CHARS", "parse_selected_solutions"}
_tree = ast.parse(_SOURCE)
_nodes = [
    node
    for node in _tree.body
    if (isinstance(node, ast.FunctionDef) and node.name in _WANTED)
    or (
        isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in _WANTED for t in node.targets
        )
    )
]
assert len(_nodes) == 2, f"expected constant + parser, found {len(_nodes)}"

_ns = {"re": re, "logger": logging.getLogger(__name__), "List": list}
exec(compile(ast.Module(body=_nodes, type_ignores=[]), str(_IDEATION), "exec"), _ns)

MIN_CHARS = _ns["MIN_SELECTED_SOLUTION_CHARS"]
parse_selected_solutions = _ns["parse_selected_solutions"]

FULL = "x" * 400


def test_rejects_degenerate_bodies():
    # Contest 5 (2026-08-06): the selector emitted "and" for slots 1-2; they
    # parsed as valid solutions and the round silently ran 2 lanes of 8.
    out = "".join(f"<solution_{i}>and</solution_{i}>" for i in (1, 2))
    assert parse_selected_solutions(out, 8) == []


def test_keeps_full_specs_only():
    out = (
        f"<solution_1>{FULL}</solution_1>"
        "<solution_2>too short</solution_2>"
        f"<solution_3>{FULL}</solution_3>"
    )
    assert parse_selected_solutions(out, 3) == [FULL, FULL]


def test_k1_contract_and_floor():
    assert parse_selected_solutions(f"<solution>{FULL}</solution>", 1) == [FULL]
    assert parse_selected_solutions("<solution>and</solution>", 1) == []


def test_floor_value_is_spec_sized():
    # Real specs ran 2,000-4,400 chars; the floor sits well below that and
    # well above a fragment.
    assert 50 < MIN_CHARS < 1000


def test_short_parse_retries_then_tops_up_from_the_pool():
    # The round must never shrink: retry once, then fill K from the pool.
    assert "retrying once" in _SOURCE
    assert "topping up from the pooled" in _SOURCE


def test_crash_retries_on_a_fallback_model():
    # Contest 5: the provider's safety classifier killed both lanes' codex
    # sessions; a crashed session must retry on the configured fallback.
    implementation_source = (
        _IDEATION.parent / "implementation.py"
    ).read_text()
    assert "implementation_fallback_model" in implementation_source
    assert "Retrying the implementation on " in implementation_source
