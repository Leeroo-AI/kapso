"""Cost admission protects the real frame evaluator and preserves refusal evidence."""

import json
import shlex
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.feedback_flow import (
    measured_cost_from_output,
)
from kapso.execution.search_strategies.generic.implementation import (
    SuspendedSession,
    build_implementation_prompt,
)
from kapso.execution.search_strategies.generic.registered_evaluation import (
    candidate_environment_command,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch


class CandidateWorkspace:
    def __init__(self, folders):
        self.folders = folders

    @contextmanager
    def materialize_ref(self, ref):
        yield str(self.folders[ref])


class FeedbackProvider:
    configured_timeout_seconds = None

    def __init__(self):
        self.outputs = []

    def generate(self, **kwargs):
        self.outputs.append(kwargs["evaluation_result"])
        return SimpleNamespace(
            feedback="Evaluation succeeded", evaluation_valid=True,
            score=0.8, stop=False, duration_seconds=None,
        )


def candidate_strategy(tmp_path, count=1):
    folders = {}
    nodes = []
    for node_id in range(count):
        folder = tmp_path / str(node_id)
        evaluation = folder / "kapso_evaluation"
        evaluation.mkdir(parents=True)
        (evaluation / "evaluate.py").write_text(
            "from pathlib import Path\n"
            "Path('executed').write_text('yes')\n"
            "print('KAPSO_EVAL_MANIFEST {\"cost\": \"3 credits\"}')\n"
        )
        ref = f"candidate-{node_id}"
        folders[ref] = folder
        nodes.append(SearchNode(
            node_id=node_id, solution="candidate", branch_name=ref,
            parent_branch_name="main", code_diff="candidate changes",
        ))
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.cost_config = {"enabled": True, "unit": "credits", "weight": 1.0}
    strategy.node_history = []
    strategy.workspace = CandidateWorkspace(folders)
    strategy.workspace_dir = str(tmp_path)
    strategy.feedback_generator = FeedbackProvider()
    strategy.goal = "improve"
    strategy.design_axes = []
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)
    strategy.registered_evaluation_manifest = {}
    strategy.registered_evaluation_command = ""
    strategy.registered_evaluator_id = None
    strategy.registered_subsample_seed = 1337
    strategy.registered_data_manifest = {}
    strategy.evaluation_provenance = "agent_generated"
    strategy.fidelity_decision = None
    strategy.record_eval_duration = None
    strategy.implementation_timeout = None
    strategy.budget_snapshot = None
    strategy.parent_policy = "best"
    strategy.env_strip = []
    strategy.env_defaults = {}
    strategy.expansion_lane_env = None
    return strategy, nodes, folders


def test_refusal_skips_evaluation_and_feedback_but_records_candidate(tmp_path):
    strategy, nodes, folders = candidate_strategy(tmp_path)

    def refuse(node):
        node.estimated_cost = {"amount": 8.0, "unit": "credits"}
        node.cost_refusal = {"cap_per_candidate": 5.0}
        node.score = None
        node.evaluation_valid = False
        return False

    strategy.before_candidate_evaluation = refuse
    strategy._finalize_nodes(nodes, time.monotonic(), append=True)
    assert strategy.node_history == nodes
    assert nodes[0].cost_refusal == {"cap_per_candidate": 5.0}
    assert not (folders[nodes[0].branch_name] / "executed").exists()
    assert strategy.feedback_generator.outputs == []
    assert strategy._select_parent().branch_name == "main"


def test_admission_runs_frame_evaluation_and_measures_before_next_lane(tmp_path):
    strategy, nodes, folders = candidate_strategy(tmp_path, count=2)
    seen = []

    def admit(node):
        seen.append([previous.measured_cost for previous in strategy.node_history])
        node.estimated_cost = {"amount": 1.0, "unit": "credits"}
        return True

    strategy.before_candidate_evaluation = admit
    strategy._finalize_nodes(nodes, time.monotonic(), append=True)
    assert seen == [[], [{"amount": 3.0, "unit": "credits"}]]
    assert all((folder / "executed").exists() for folder in folders.values())
    assert all(node.score == 0.8 for node in nodes)
    assert len(strategy.feedback_generator.outputs) == 2
    assert all("KAPSO_EVAL_MANIFEST" in output for output in strategy.feedback_generator.outputs)


@pytest.mark.parametrize("cost", [3.0, "3 credits"])
def test_disabled_cost_preserves_feedback_without_measuring_cost(tmp_path, cost):
    strategy, nodes, _ = candidate_strategy(tmp_path)
    strategy.cost_config["enabled"] = False
    nodes[0].evaluation_output = "KAPSO_EVAL_MANIFEST " + json.dumps({"cost": cost})

    strategy._finalize_nodes(nodes, time.monotonic(), append=True)

    assert nodes[0].score == 0.8
    assert nodes[0].evaluation_valid is True
    assert nodes[0].measured_cost is None
    assert strategy.node_history == nodes


def admit_estimate(node):
    node.estimated_cost = {"amount": 1.0, "unit": "credits"}
    return True


def test_deferred_evaluation_preserves_configured_session_environment(tmp_path, monkeypatch):
    strategy, nodes, folders = candidate_strategy(tmp_path, count=2)
    monkeypatch.setenv("KAPSO_TEST_AMBIENT", "ambient")
    monkeypatch.setenv("KAPSO_TEST_STRIPPED", "remove")
    monkeypatch.setenv("KAPSO_TEST_EMPTY", "")
    monkeypatch.setenv("KAPSO_TEST_CREDENTIAL", "inherited-token")
    monkeypatch.delenv("KAPSO_TEST_DEFAULT", raising=False)
    literal = "literal $(touch injected); 'quoted' $value"
    strategy.env_strip = ["KAPSO_TEST_STRIPPED"]
    strategy.env_defaults = {
        "KAPSO_TEST_AMBIENT": "default",
        "KAPSO_TEST_DEFAULT": literal,
        "KAPSO_TEST_EMPTY": "default",
        "KAPSO_TEST_LANE": "default",
    }
    strategy.expansion_lane_env = [
        {"KAPSO_TEST_LANE": "lane-zero"}, {"KAPSO_TEST_LANE": "lane-one"},
    ]
    names = [
        "KAPSO_TEST_AMBIENT", "KAPSO_TEST_STRIPPED", "KAPSO_TEST_EMPTY",
        "KAPSO_TEST_CREDENTIAL", "KAPSO_TEST_DEFAULT", "KAPSO_TEST_LANE",
    ]
    for folder in folders.values():
        (folder / "kapso_evaluation" / "evaluate.py").write_text(
            "import json, os\n"
            f"print(json.dumps({{name: os.environ.get(name) for name in {names!r}}}))\n"
        )
    strategy.before_candidate_evaluation = admit_estimate
    strategy._finalize_nodes(nodes, time.monotonic(), append=True)
    for node, lane in zip(nodes, ["lane-zero", "lane-one"]):
        assert json.loads(node.evaluation_output) == {
            "KAPSO_TEST_AMBIENT": "ambient", "KAPSO_TEST_STRIPPED": None,
            "KAPSO_TEST_EMPTY": "", "KAPSO_TEST_CREDENTIAL": "inherited-token",
            "KAPSO_TEST_DEFAULT": literal, "KAPSO_TEST_LANE": lane,
        }
        assert not (folders[node.branch_name] / "injected").exists()


def test_deferred_evaluation_rejects_shell_syntax_in_environment_names():
    with pytest.raises(ValueError, match="environment variable name"):
        candidate_environment_command(
            [sys.executable, "kapso_evaluation/evaluate.py"], env_strip=[],
            env_defaults={"VALUE; echo injected": "value"}, env_overrides=None,
        )


def test_finalized_candidate_output_replaces_implementation_claims(tmp_path):
    strategy, nodes, _ = candidate_strategy(tmp_path)
    strategy.before_candidate_evaluation = admit_estimate
    nodes[0].evaluation_output = 'KAPSO_EVAL_MANIFEST {"cost": "999 credits"}'
    nodes[0].score = 999
    strategy._finalize_nodes(nodes, time.monotonic(), append=True)
    assert nodes[0].measured_cost == {"amount": 3.0, "unit": "credits"}
    assert nodes[0].score == 0.8
    assert "999" not in strategy.feedback_generator.outputs[0]


def test_registered_manifest_keeps_raw_score_and_records_measured_cost(tmp_path):
    strategy, nodes, _ = candidate_strategy(tmp_path)
    evaluation = tmp_path / "kapso_evaluation"
    evaluation.mkdir()
    manifest = {
        "fidelity": "full", "fraction": 1.0, "seed": 1337,
        "items": 1, "total_items": 1, "score": 0.95, "cost": "3 credits",
    }
    (evaluation / "evaluate.py").write_text(
        f"print({'KAPSO_EVAL_MANIFEST ' + json.dumps(manifest)!r})\n"
    )
    strategy.registered_evaluation_command = (
        f"{shlex.quote(sys.executable)} kapso_evaluation/evaluate.py"
    )
    finalized = []
    strategy.problem_handler.finalize_run_selection = (
        lambda record, valid: finalized.append((record, valid))
    )
    strategy.before_candidate_evaluation = admit_estimate
    strategy._finalize_nodes(nodes, time.monotonic(), append=True)
    assert nodes[0].score == 0.95
    assert nodes[0].measured_cost == {"amount": 3.0, "unit": "credits"}
    assert finalized == [(manifest, True)]


@pytest.mark.parametrize("failure", ["exit", "deadline"])
def test_failed_evaluator_cannot_be_revalidated_by_feedback(tmp_path, failure):
    strategy, nodes, folders = candidate_strategy(tmp_path)
    script = folders[nodes[0].branch_name] / "kapso_evaluation" / "evaluate.py"
    script.write_text(script.read_text() + (
        "raise SystemExit(1)\n" if failure == "exit"
        else "import sys, time\nsys.stdout.flush()\ntime.sleep(60)\n"
    ))
    if failure == "deadline":
        strategy.implementation_timeout = 0.5
    strategy.before_candidate_evaluation = admit_estimate
    strategy._finalize_nodes(nodes, time.monotonic(), append=True)
    assert nodes[0].evaluation_valid is False
    assert nodes[0].score is None
    assert nodes[0].should_stop is False
    assert nodes[0].measured_cost == {"amount": 3.0, "unit": "credits"}
    assert strategy.node_history == nodes
    assert strategy.feedback_generator.outputs == []


def test_inconsistent_measured_unit_fails_before_feedback(tmp_path):
    strategy, nodes, folders = candidate_strategy(tmp_path)
    script = folders[nodes[0].branch_name] / "kapso_evaluation" / "evaluate.py"
    script.write_text(script.read_text().replace("3 credits", "3 euros"))
    strategy.before_candidate_evaluation = admit_estimate
    with pytest.raises(ValueError, match="does not match cost.unit"):
        strategy._finalize_nodes(nodes, time.monotonic(), append=True)
    assert strategy.feedback_generator.outputs == []


def test_enabled_initialization_requires_estimator_before_workspace_setup(tmp_path):
    with pytest.raises(ValueError, match="cost.estimator is required"):
        GenericSearch(SimpleNamespace(params={"cost": {"enabled": True}}), str(tmp_path))


def test_suspended_build_cannot_publish_an_unestimated_score(tmp_path):
    strategy, nodes, folders = candidate_strategy(tmp_path)
    output = (
        "<score>999</score><evaluation_output>"
        'KAPSO_EVAL_MANIFEST {"cost": "9 credits"}'
        "</evaluation_output>"
    )
    strategy._absorb_implementation(
        nodes[0], output, None, SuspendedSession([1], "session"),
    )
    strategy._finalize_nodes(nodes, time.monotonic(), append=True)
    assert nodes[0].suspended is True
    assert nodes[0].score is None
    assert nodes[0].evaluation_valid is False
    assert nodes[0].evaluation_output == ""
    assert nodes[0].measured_cost is None
    assert nodes[0].estimated_cost is None
    assert strategy.get_best_experiment() is None
    assert strategy.get_experiment_history(best_last=True) == nodes
    assert strategy._select_parent().branch_name == "main"
    assert not (folders[nodes[0].branch_name] / "executed").exists()


def test_deferred_evaluator_rejects_missing_or_refused_admission(tmp_path):
    strategy, nodes, folders = candidate_strategy(tmp_path)
    with pytest.raises(ValueError, match="admitted cost estimate"):
        strategy.evaluate_candidate(nodes[0])
    admit_estimate(nodes[0])
    nodes[0].cost_refusal = {"reason": "cap_per_candidate"}
    with pytest.raises(ValueError, match="admitted cost estimate"):
        strategy.evaluate_candidate(nodes[0])
    assert not (folders[nodes[0].branch_name] / "executed").exists()


def test_enabled_strategy_requires_admission_callback(tmp_path):
    strategy, nodes, _ = candidate_strategy(tmp_path)
    strategy.before_candidate_evaluation = None
    with pytest.raises(ValueError, match="before_candidate_evaluation"):
        strategy._finalize_nodes(nodes, time.monotonic(), append=True)


def test_cost_winner_is_parent_and_deliverable_with_registered_evaluation(tmp_path):
    strategy, nodes, _ = candidate_strategy(tmp_path, count=2)
    for node, amount in zip(nodes, [8.0, 2.0]):
        node.score = 0.8
        node.estimated_cost = {"amount": amount, "unit": "credits"}
    strategy.node_history = nodes
    strategy.registered_evaluator_id = "registered-head"
    assert strategy.get_best_experiment() is nodes[1]
    assert strategy._select_parent().node_id == nodes[1].node_id
    assert strategy.get_deliverable_experiment() is nodes[1]
    assert strategy.get_experiment_history(best_last=True) == nodes


@pytest.mark.parametrize("cost", ["-1 credits", "nan credits", "1", 3])
def test_malformed_measured_cost_fails_loud(cost):
    with pytest.raises(ValueError, match="cost"):
        measured_cost_from_output("KAPSO_EVAL_MANIFEST " + json.dumps({"cost": cost}))


def test_measured_cost_uses_last_manifest_and_keeps_full_unit():
    output = '\n'.join([
        'KAPSO_EVAL_MANIFEST {"cost": "10 credits"}',
        'KAPSO_EVAL_MANIFEST {"cost": "2.5 core hours"}',
    ])
    assert measured_cost_from_output(output) == {"amount": 2.5, "unit": "core hours"}


def test_final_manifest_without_cost_does_not_reuse_an_earlier_measurement():
    output = '\n'.join([
        'KAPSO_EVAL_MANIFEST {"cost": "10 credits"}',
        'KAPSO_EVAL_MANIFEST {"score": 0.9}',
    ])
    assert measured_cost_from_output(output) is None


def test_corrupt_cost_manifest_fails_loud():
    with pytest.raises(json.JSONDecodeError):
        measured_cost_from_output('KAPSO_EVAL_MANIFEST {"cost":')


def test_cost_build_prompt_defers_all_evaluation_to_orchestrator():
    prompt = build_implementation_prompt(
        solution="full solution", problem="full problem", branch_name="candidate",
        repo_memory_brief="memory", repo_memory_detail_access_instructions="details",
        previous_errors="", budget_status="budget", evaluation_instructions="RUN EVALUATION NOW",
        shared_artifacts_brief="cache", defer_evaluation=True,
    )
    assert "Do not run the candidate or evaluation" in prompt
    assert "RUN EVALUATION NOW" not in prompt
    assert "<score>null</score>" in prompt
    assert "full solution" in prompt and "full problem" in prompt
