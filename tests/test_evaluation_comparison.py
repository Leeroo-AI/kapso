"""Baseline reporting does not leak holdout metrics into search selection."""

import pytest

from kapso.execution.iteration_evaluator import (
    IterationEvaluationResult,
    IterationEvaluationValidationError,
    normalize_result,
    comparison_summary,
)
from kapso.execution.observability import EvolveStatus, OperationStatusView


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


def test_status_without_baseline_has_no_external_comparison(tmp_path):
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


@pytest.mark.parametrize(
    "report,error", [("invalid", TypeError), ({}, KeyError)]
)
def test_malformed_comparison_raises(report, error):
    with pytest.raises(error):
        comparison_summary({"baseline_comparison": report})


def test_misnamed_baseline_is_rejected():
    with pytest.raises(IterationEvaluationValidationError, match="acc"):
        normalize_result(
            IterationEvaluationResult(
                metrics={"accuracy": 0.9}, baseline_metrics={"acc": 0.8}
            )
        )


def test_conflicting_reserved_metadata_is_rejected():
    with pytest.raises(IterationEvaluationValidationError, match="reserved"):
        normalize_result(
            IterationEvaluationResult(
                metrics={"accuracy": 0.9},
                baseline_metrics={"accuracy": 0.8},
                metadata={"baseline_comparison": {"forged": True}},
            )
        )
