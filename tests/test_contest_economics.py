"""Reward-shaping contract: contest economics + allocation grading.

Pins the anti-play-safe design in all four homes — the IOAI handler renders
the config-sourced economics (single priced insurance point, gain-ratio-gated
confirmations, explicit final-score-only reward with the boldness permission),
the old always-ship-safely instruction is gone, and the core templates carry
the critical-path/time-allocation recon step, the feedback allocation-grading
axes, and the selector's time-to-first-strong-score rubric.
"""

import time
from pathlib import Path

import pytest

from kapso.core.prompt_loader import load_prompt
from benchmarks.ioai.handler import AnimalDeductionHandler

ECONOMICS = {"insurance_minutes": 5, "confirm_gain_ratio": 2.0}
CAPS = {"ideation_timeout": 1080, "implementation_timeout": 5400}


def make_handler(tmp_path, **overrides):
    kwargs = dict(
        task_dir=str(tmp_path),
        statement="STMT",
        deadline_ts=time.time() + 7200,
        session_caps=CAPS,
        contest_economics=ECONOMICS,
    )
    kwargs.update(overrides)
    return AnimalDeductionHandler(**kwargs)


def test_handler_renders_economics_from_config_values(tmp_path):
    context = make_handler(tmp_path).get_problem_context(budget_progress=10)
    assert "rewarded ONLY for the final frozen submission" in context
    assert "Bold-and-correct beats" in context
    assert "~5 minutes" in context           # insurance_minutes reaches the prompt
    assert "2× what the confirmation" in context  # confirm_gain_ratio reaches it
    assert "never skip it" in context        # freeze confirm stays mandatory
    # The old instructed-safety line must never come back.
    assert "beats an empty directory" not in context


def test_handler_requires_the_economics_knobs(tmp_path):
    with pytest.raises(ValueError, match="contest_economics"):
        make_handler(tmp_path, contest_economics={"insurance_minutes": 5})
    with pytest.raises(ValueError, match="contest_economics"):
        make_handler(tmp_path, contest_economics=None)


def test_core_templates_carry_the_allocation_contract():
    implementation = load_prompt(
        "execution/search_strategies/generic/prompts/implementation_claude_code.md"
    )
    assert "Identify the critical path" in implementation
    assert "TIME ALLOCATION" in implementation

    feedback = load_prompt(
        "execution/search_strategies/generic/feedback_generator/prompts/feedback_generator.md"
    )
    assert "time-to-first-strong-score" in feedback
    assert "critical-path velocity" in feedback
    assert "counterfactual ledger" in feedback
    assert "never to be traded for speed" in feedback

    selector = load_prompt(
        "execution/search_strategies/generic/prompts/ideation_selector.md"
    )
    assert "Time-to-first-strong-score" in selector
    assert "staged plan beats a monolithic one" in selector
