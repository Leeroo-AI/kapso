"""Baseline reporting does not leak holdout metrics into search selection."""

import json

import pytest

from kapso.execution.iteration_evaluator import (
    IterationEvaluationResult,
    IterationEvaluationValidationError,
    normalize_result,
)
from kapso.execution.evaluation_comparison import comparison_summary
from kapso.execution.observability import EvolveStatus, OperationStatusView
from kapso.execution.run_checkpoint import RunCheckpointStore
from test_iteration_evaluator import _init_workspace, _orchestrator


def test_comparison_directions_missing_baseline_and_unspecified_direction():
    result = normalize_result(
        IterationEvaluationResult(
            metrics={"accuracy": 0.9, "latency": 8, "cost": 2, "new": 1},
            baseline_metrics={"accuracy": 0.8, "latency": 10, "cost": 1},
            metric_directions={"accuracy": "maximize", "latency": "minimize"},
        )
    )
    report = result.metadata["baseline_comparison"]
    assert report["metrics"]["accuracy"]["delta"] == pytest.approx(0.1)
    assert report["metrics"]["accuracy"]["verdict"] == "improved"
    assert report["metrics"]["latency"]["delta"] == -2
    assert report["metrics"]["latency"]["verdict"] == "improved"
    assert report["metrics"]["cost"]["verdict"] == "unclassified"
    assert report["missing_baseline"] == ["new"]


@pytest.mark.parametrize(
    "value,verdict", [(0.7, "regressed"), (0.8, "unchanged")]
)
def test_regression_and_tie(value, verdict):
    result = normalize_result(
        IterationEvaluationResult(
            metrics={"accuracy": value},
            baseline_metrics={"accuracy": 0.8},
            metric_directions={"accuracy": "maximize"},
        )
    )
    assert (
        result.metadata["baseline_comparison"]["metrics"]["accuracy"][
            "verdict"
        ]
        == verdict
    )


@pytest.mark.parametrize("direction", ["unknown", [], None])
def test_invalid_direction_is_rejected(direction):
    with pytest.raises(
        IterationEvaluationValidationError, match="metric_directions"
    ):
        normalize_result(
            IterationEvaluationResult(
                metrics={"accuracy": 0.9},
                metric_directions={"accuracy": direction},
            )
        )


def test_comparison_persists_with_lineage_without_changing_selection(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "campaign"
    _init_workspace(workspace)

    def evaluator(context):
        return IterationEvaluationResult(
            metrics={"accuracy": 0.9 - context.node.node_id * 0.8},
            baseline_metrics={"accuracy": 0.5},
            metric_directions={"accuracy": "maximize"},
        )

    orchestrator = _orchestrator(workspace, monkeypatch, evaluator)
    result = orchestrator.solve(experiment_max_iter=1)
    # Internal scores still select node 1, despite its external regression.
    assert result.best_experiment.node_id == 1
    history = json.loads(
        (workspace / ".kapso" / "experiment_history.json").read_text()
    )
    checkpoint = RunCheckpointStore(str(workspace)).load()
    nodes = checkpoint.strategy_state["node_history"]
    for record, node in zip(history, nodes):
        assert (
            record["external_evaluation_metadata"]
            == node["external_evaluation_metadata"]
        )
    assert (
        nodes[1]["external_evaluation_metadata"]["baseline_comparison"][
            "metrics"
        ]["accuracy"]["verdict"]
        == "regressed"
    )
    restored = OperationStatusView(workspace)
    assert "regressed" in restored.explain()
    assert "external node 1" in restored.explain()
    assert "improved" in restored.explain(tree=True)
    assert "regressed" in restored.explain(tree=True)


def test_legacy_status_has_no_external_comparison(tmp_path):
    path = tmp_path / "status.json"
    EvolveStatus(path).done()
    assert "external" not in OperationStatusView(path).explain()


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), "0.5"])
def test_invalid_baseline_is_rejected(value):
    with pytest.raises(IterationEvaluationValidationError):
        normalize_result(
            IterationEvaluationResult(
                metrics={"accuracy": 0.9},
                baseline_metrics={"accuracy": value},
            )
        )


def test_normalization_is_idempotent_and_detaches_baseline():
    baseline = {"accuracy": 0.8}
    result = normalize_result(
        IterationEvaluationResult(
            metrics={"accuracy": 0.9},
            baseline_metrics=baseline,
            metric_directions={"accuracy": "maximize"},
        )
    )
    baseline["accuracy"] = 0
    assert normalize_result(result) == result
    assert result.baseline_metrics["accuracy"] == 0.8


@pytest.mark.parametrize("report", ["legacy", {"metrics": {"x": None}}])
def test_legacy_metadata_cannot_crash_watch(report):
    assert comparison_summary({"baseline_comparison": report}) == ""


def test_resume_preserves_baseline_reports(tmp_path, monkeypatch):
    workspace = tmp_path / "campaign"
    _init_workspace(workspace)
    calls = []

    def evaluator(context):
        calls.append((context.iteration, context.node.node_id))
        return IterationEvaluationResult(
            metrics={"accuracy": context.node.node_id / 10},
            baseline_metrics={"accuracy": 0.2},
            metric_directions={"accuracy": "maximize"},
            metadata={"baseline_ref": "fixed-baseline"},
        )

    _orchestrator(workspace, monkeypatch, evaluator).solve(
        experiment_max_iter=1
    )
    resumed = _orchestrator(workspace, monkeypatch, evaluator, resume=True)
    resumed.solve(experiment_max_iter=1)
    assert calls == [(1, 0), (1, 1), (2, 2), (2, 3)]
    nodes = (
        RunCheckpointStore(str(workspace))
        .load()
        .strategy_state["node_history"]
    )
    verdicts = [
        node["external_evaluation_metadata"]["baseline_comparison"]["metrics"][
            "accuracy"
        ]["verdict"]
        for node in nodes
    ]
    assert verdicts == ["regressed", "regressed", "unchanged", "improved"]


def test_conflicting_reserved_metadata_is_rejected():
    with pytest.raises(IterationEvaluationValidationError, match="reserved"):
        normalize_result(
            IterationEvaluationResult(
                metrics={"accuracy": 0.9},
                baseline_metrics={"accuracy": 0.8},
                metadata={"baseline_comparison": {"forged": True}},
            )
        )
