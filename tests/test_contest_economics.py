"""Insured finalization + allocation-grading contract (platform side).

Pins the insured iteration-admission floor (BudgetSpec), the base handler's
uninsured default, and the core templates' critical-path/time-allocation
duties. The benchmark-side half (a handler banking a confirmed score to
shrink its reserve) is pinned by the ioai2026 benchmark suite; the retired
reward-sermon generation and its Animal Deduction pipe-cleaner benchmark
were deleted in the platform unification (decision #16 and the ioai2026
restructure).
"""

import pytest

from kapso.core.prompt_loader import load_prompt


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
