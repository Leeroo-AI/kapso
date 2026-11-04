"""Each parallel lane's judge reads how ITS OWN build session ended.

The feedback judge decides validity partly from the harness's account of
the session's end (died, killed at its deadline, completed). Lanes of one
expansion round run concurrently, so that account must be kept per node:
judging node A on node B's ending turns a crash into a clean finish, or
a clean finish into a kill.
"""

from types import SimpleNamespace

import kapso.execution.search_strategies.generic.strategy as strategy_module
from kapso.core.roles import SessionRole
from kapso.execution.search_strategies.generic.strategy import GenericSearch


ENDINGS = {
    "generic_exp_0": "implementation session died prematurely (provider kill)",
    "generic_exp_1": "implementation session ended naturally",
}


def test_each_lane_is_judged_on_its_own_session_ending(monkeypatch, tmp_path):
    judged = {}

    def fake_run_implementation(**kwargs):
        kwargs["note_session_end_facts"](ENDINGS[kwargs["branch_name"]])
        return "<score>1</score>", {}, None, None, None

    class Judge:
        configured_timeout_seconds = 60
        agent_type = "claude_code"

        def attach_harness(self, staged):
            """No harness here: nothing to attach."""

        def generate(self, **kwargs):
            judged[kwargs["head_branch"]] = kwargs["session_end_facts"]
            return SimpleNamespace(
                feedback="ok", evaluation_valid=True, score=1.0, stop=False,
                duration_seconds=None, cost_usd=0.0, cards_load_bearing=[], unjudged="",
            )

    monkeypatch.setattr(strategy_module, "run_implementation", fake_run_implementation)
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.iteration_count = 1
    strategy.harness = None
    strategy.__dict__.update(
        node_expansion_value=2, expansion_lane_env=None, node_history=[],
        budget_snapshot=None, budget_snapshot_monotonic=None, finishing_cut=None,
        workspace=None, workspace_dir=str(tmp_path), llm=None,
        registered_evaluation_manifest=None, registered_evaluation_command=None,
        registered_evaluator_id=None, implementation_gates=[],
        gate_failure_policy="skip", bank_serving=None, kg_index_path=None, kg_config_path=None, hosted=None,
        implementation_role=SessionRole(name="implementation", cli="claude_code", model="m", effort=None, timeout=600, web=False),
        implementation_fallback_model=None, env_strip=[], env_defaults={},
        previous_errors=[], recent_error_count=3, feedback_generator=Judge(),
        goal="the goal",
        problem_handler=SimpleNamespace(maximize_scoring=True),
        _session_end_facts={},
    )
    monkeypatch.setattr(strategy, "_status_phase", lambda phase: None)
    monkeypatch.setattr(strategy, "_clamped_timeout", lambda seconds: seconds)
    monkeypatch.setattr(strategy, "_get_code_diff", lambda branch, parent: "diff")
    monkeypatch.setattr(strategy, "_attempt_commit", lambda branch_name: "c" * 40)
    monkeypatch.setattr(strategy, "_ensure_technical_difficulties", lambda node: None)
    monkeypatch.setattr(strategy, "enforce_evaluation_integrity", lambda node: True)

    strategy._expand_round(
        problem="p", solutions=["a", "b"],
        parent=SimpleNamespace(node_id=None, branch_name="main"),
        decision=None, ideation_sections=[], ideation_telemetry={},
        iteration_started_at="t0", iteration_started_monotonic=0.0,
    )

    assert judged == ENDINGS
