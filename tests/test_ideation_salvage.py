"""Hermetic tests for salvaging deadline-terminated ideation output.

A live run lost 30 minutes of ideation research when the session hit its
deadline: the strategy substituted the generic fallback solution and the
partial output (which carried the research) was discarded. These pin the
salvage contract: deadline kills with substantive output become the
solution; crashes and near-empty kills keep the explicit fallback.
"""

from kapso.execution.coding_agents.base import CodingResult
from kapso.execution.search_strategies.generic.strategy import (
    GenericSearch,
    MIN_IDEATION_SALVAGE_CHARS,
)

SALVAGE_HEADER = "Salvaged from a deadline-terminated ideation session"


def make_strategy():
    return GenericSearch.__new__(GenericSearch)


def deadline_result(output):
    return CodingResult(
        success=False,
        output=output,
        error="Claude Code CLI exceeded its 1800s deadline and was terminated",
        metadata={"deadline_exceeded": True},
    )


def test_deadline_killed_ideation_output_is_salvaged():
    research = (
        "Compared GSM8K-safe training sets.\n"
        "# Core Idea\n"
        "SFT on OpenMathInstruct-2 rendered with the eval chat template."
        + " More detail." * 30
    )
    salvaged = make_strategy()._salvage_ideation_output(
        deadline_result(research)
    )
    assert salvaged is not None
    assert SALVAGE_HEADER in salvaged
    assert "OpenMathInstruct-2" in salvaged


def test_salvage_prefers_tagged_solution_content():
    output = ("tool noise " * 30) + "<solution># Plan\nTrain LoRA</solution>"
    salvaged = make_strategy()._salvage_ideation_output(
        deadline_result(output)
    )
    assert salvaged is not None
    assert "# Plan\nTrain LoRA" in salvaged
    assert "tool noise" not in salvaged


def test_near_empty_partial_output_is_not_salvaged():
    result = deadline_result("x" * (MIN_IDEATION_SALVAGE_CHARS - 1))
    assert make_strategy()._salvage_ideation_output(result) is None


def test_non_deadline_failures_are_not_salvaged():
    result = CodingResult(
        success=False,
        output="Traceback (most recent call last): boom " * 20,
        error="CLI exited with code 1",
        metadata={},
    )
    assert make_strategy()._salvage_ideation_output(result) is None
