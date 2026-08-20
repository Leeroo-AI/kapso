"""Insured finalization + allocation-grading contract.

Pins the surviving generation of the contest-economics design: the IOAI
handler's insured reserve (a confirmed banked score shrinks the endgame
reserve), the platform's insured iteration-admission floor, and the core
templates' critical-path/time-allocation recon step, feedback
allocation-grading axes, and selector time-to-first-strong-score rubric.
The retired reward-sermon generation (insurance_minutes/confirm_gain_ratio
prompt economics) was deleted in the platform unification (decision #16).
"""

import time
from pathlib import Path

import pytest

from kapso.core.prompt_loader import load_prompt
from benchmarks.ioai.handler import AnimalDeductionHandler

INSURED_RESERVE_SECONDS = 240.0
CAPS = {"ideation_timeout": 1080, "implementation_timeout": 5400}


def make_handler(tmp_path, **overrides):
    kwargs = dict(
        task_dir=str(tmp_path),
        statement="STMT",
        deadline_ts=time.time() + 7200,
        session_caps=CAPS,
        insured_reserve_seconds=INSURED_RESERVE_SECONDS,
    )
    kwargs.update(overrides)
    return AnimalDeductionHandler(**kwargs)


def test_handler_requires_a_positive_insured_reserve(tmp_path):
    with pytest.raises(ValueError, match="insured_reserve_seconds"):
        make_handler(tmp_path, insured_reserve_seconds=0)
    with pytest.raises(ValueError, match="insured_reserve_seconds"):
        make_handler(tmp_path, insured_reserve_seconds=None)


def test_handler_prompt_carries_no_retired_economics_sermon(tmp_path):
    # The reward/insurance/gain-ratio prompt economics are a retired design
    # generation (kaggle dropped them for the single budget knob; decision
    # #16 removed them here too). The operational submission contract
    # (atomic swap, promote only on confirmed wins) lives on.
    context = make_handler(tmp_path).get_problem_context(budget_progress=10)
    assert "Reward & time economics" not in context
    assert "insurance" not in context.lower()
    assert "beats an empty directory" not in context
    assert "atomically" in context


def test_insured_reserve_requires_confirmed_nonplaceholder_score(tmp_path):
    handler = make_handler(tmp_path)
    # Nothing on disk yet.
    assert handler.deliverable_ready_reserve_seconds() is None
    # Solution alone is not confirmation.
    Path(handler.submission_dir, "solution.py").write_text("class MySolution: ...")
    assert handler.deliverable_ready_reserve_seconds() is None
    # Placeholder insurance banks 0.0 — still not insured.
    log = Path(handler.task_dir, "best_score.log")
    log.write_text("0.0 2026-07-21T11:46:30Z trivial-random-guess\n")
    assert handler.deliverable_ready_reserve_seconds() is None
    # A confirmed real score flips it to the freeze residual.
    log.write_text(
        "0.0 2026-07-21T11:46:30Z trivial-random-guess\n"
        "0.7341 2026-07-21T12:32:23Z bsc-infogain-72cols\n"
    )
    assert handler.deliverable_ready_reserve_seconds() == INSURED_RESERVE_SECONDS
    # A corrupt score line raises — never silently uninsured.
    log.write_text("not-a-score whatever\n")
    with pytest.raises(ValueError):
        handler.deliverable_ready_reserve_seconds()


def test_budget_spec_insured_floor():
    from kapso.execution.budget import BudgetSpec

    spec = BudgetSpec.resolve(
        config_block={
            "min_iteration_seconds": 900,
            "min_iteration_seconds_insured": 300,
        },
        time_budget_minutes=120,
    )
    assert spec.effective_min_iteration_seconds(insured=False) == 900
    assert spec.effective_min_iteration_seconds(insured=True) == 300
    assert spec.to_dict()["min_iteration_seconds_insured"] == 300

    plain = BudgetSpec.resolve(config_block={"min_iteration_seconds": 900})
    assert plain.effective_min_iteration_seconds(insured=True) == 900

    with pytest.raises(ValueError):
        BudgetSpec(min_iteration_seconds_insured=-1)


def test_base_handler_defaults_to_uninsured():
    from kapso.environment.handlers.base import ProblemHandler

    class Minimal(ProblemHandler):
        def get_problem_context(self, budget_progress: float = 0, **kwargs) -> str:
            return "ctx"

    assert Minimal().deliverable_ready_reserve_seconds() is None


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
