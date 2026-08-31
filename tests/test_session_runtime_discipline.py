"""Single-source guard for the session-runtime discipline block.

The watcher/alarm/notification/kill rules are runtime truths of Claude Code
sessions, not benchmark facts: they live ONLY in the core implementation
template. This pins both directions of the contract — the core template
carries them, and benchmark handlers do not re-grow drifting copies (the
IOAI run-1 idle-block happened exactly because a trimmed handler copy
dropped the dead-man's-alarm rule).
"""

from pathlib import Path

from kapso.core.prompt_loader import load_prompt

DISCIPLINE_MARKERS = [
    "DEAD-MAN'S ALARM",
    "WATCHER DISCIPLINE",
    "KILL DISCIPLINE",
    "completion notification is EVIDENCE",
    "NO ORPHANED VALUE",
]

# Every benchmark handler in the tree, discovered rather than listed: a new
# benchmark must inherit the guard without anyone remembering to add it, and
# the guard must not vanish when a benchmark leaves (posttrain — the single
# path this used to name — went untracked 2026-08-31). The incident that
# motivated it was an IOAI handler, so a one-benchmark list was never the
# right shape anyway.
HANDLER_PATHS = sorted(
    (Path(__file__).parent.parent / "benchmarks").rglob("handler.py")
)


def test_core_template_carries_the_discipline():
    template = load_prompt(
        "execution/search_strategies/generic/prompts/implementation_claude_code.md"
    )
    for marker in DISCIPLINE_MARKERS:
        assert marker in template, f"core template lost: {marker}"


def test_handlers_do_not_duplicate_the_discipline():
    for path in HANDLER_PATHS:
        source = path.read_text(encoding="utf-8")
        for marker in DISCIPLINE_MARKERS:
            assert marker not in source, (
                f"{path.name} re-grew a copy of {marker!r} — it belongs "
                "only in the core implementation template"
            )
