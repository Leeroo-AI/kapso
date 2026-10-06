"""Synthetic baseline reporting; no APIs, agents or benchmarks needed."""

from kapso.execution.evaluation_comparison import comparison_summary
from kapso.execution.iteration_evaluator import (
    IterationEvaluationResult,
    normalize_result,
)


def evaluate_candidate(accuracy, latency_ms):
    """Real evaluators measure candidates and a fixed, comparable baseline."""
    return IterationEvaluationResult(
        metrics={"accuracy": accuracy, "latency_ms": latency_ms},
        baseline_metrics={"accuracy": 0.8, "latency_ms": 10.0},
        metric_directions={"accuracy": "maximize", "latency_ms": "minimize"},
        metadata={
            "suite": "synthetic-v1",
            "baseline_ref": "synthetic-baseline",
        },
    )


if __name__ == "__main__":
    for label, accuracy, latency in [
        ("better", 0.9, 8.0),
        ("tradeoff", 0.9, 12.0),
        ("unchanged", 0.8, 10.0),
    ]:
        result = normalize_result(evaluate_candidate(accuracy, latency))
        print(f"{label}: {comparison_summary(result.metadata)}")
