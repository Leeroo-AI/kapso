"""Pin the selector hardening: quality floor, and K lanes always filled."""
import sys

sys.path.insert(0, "/home/ubuntu/kapso/.claude/worktrees/ioai-2025/src")

from kapso.execution.search_strategies.generic.strategy import (  # noqa: E402
    MIN_SELECTED_SOLUTION_CHARS,
    parse_selected_solutions,
)

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
    got = parse_selected_solutions(out, 3)
    assert got == [FULL, FULL], [len(s) for s in got]


def test_k1_contract_and_floor():
    assert parse_selected_solutions(f"<solution>{FULL}</solution>", 1) == [FULL]
    assert parse_selected_solutions("<solution>and</solution>", 1) == []


def test_floor_value_is_spec_sized():
    # Real specs ran 2,000-4,400 chars; the floor must sit well below that
    # and well above a fragment.
    assert 50 < MIN_SELECTED_SOLUTION_CHARS < 1000


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
