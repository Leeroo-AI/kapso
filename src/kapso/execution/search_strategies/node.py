"""Dependency-pure search-node state contract."""

import math
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict, List, Optional

from kapso.execution.evaluation_integrity import (
    AGENT_GENERATED,
    VALID_PROVENANCE,
)
from kapso.execution.fidelity import FIDELITIES, EvaluationAttempt
from kapso.execution.iteration_evaluator import normalize_metadata, normalize_metrics


@dataclass
class SearchNode:
    """
    Unified node structure for search strategies.

    Accumulates data through the node lifecycle:
    1. Solution generation -> solution populated
    2. Implementation -> branch_name, code_changes_summary populated
    3. Evaluation -> evaluation_script_path, evaluation_output populated
    4. Feedback -> feedback, score, should_stop populated
    """

    node_id: int
    parent_node_id: Optional[int] = None
    execution_revision: int = 0

    # Required provenance for GenericSearch's evidence-directed campaign.
    # Other strategy node types do not participate in the idea archive.
    idea_id: Optional[str] = None
    selection_batch_id: Optional[str] = None

    # Step 1: Solution generation
    solution: str = ""

    # Step 2: Implementation
    branch_name: str = ""
    parent_branch_name: str = ""
    implementation_base_ref: str = ""
    diff_base_ref: str = ""
    feedback_base_ref: str = ""
    code_changes_summary: str = ""
    agent_output: str = ""

    # Step 3: Evaluation (extracted from agent output or result.json)
    evaluation_script_path: str = ""
    evaluation_output: str = ""

    # Step 4: Feedback
    feedback: str = ""
    score: Optional[float] = None
    should_stop: bool = False
    evaluation_valid: bool = True
    evaluation_provenance: str = AGENT_GENERATED
    evaluation_integrity_error: str = ""

    # Observational metrics from a caller-owned iteration evaluator.
    metrics: Dict[str, float] = field(default_factory=dict)
    primary_metric: Optional[str] = None
    external_evaluation_metadata: Dict[str, Any] = field(default_factory=dict)
    external_evaluation_error: str = ""

    # Per-iteration budget telemetry.
    duration_seconds: Optional[float] = None
    cost_usd: Optional[float] = None
    started_at: str = ""
    phase_telemetry: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Fidelity and append-only measurements.
    build_fidelity: str = "full"
    eval_fidelity: str = "full"
    promoted_from: Optional[int] = None
    evaluation_attempts: List[EvaluationAttempt] = field(default_factory=list)

    # Metadata
    had_error: bool = False
    recoverable_error: bool = False
    error_message: str = ""
    workspace_dir: str = ""
    code_diff: str = ""
    technical_difficulties: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stable base-node fields to JSON-compatible data."""
        values = {}
        for item in fields(SearchNode):
            if hasattr(self, item.name):
                values[item.name] = getattr(self, item.name)
            elif item.default is not MISSING:
                values[item.name] = item.default
            elif item.default_factory is not MISSING:
                values[item.name] = item.default_factory()
        values["evaluation_attempts"] = [
            attempt.to_dict() for attempt in values["evaluation_attempts"]
        ]
        return values

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchNode":
        """Load a node while tolerating fields added by newer versions."""
        if not isinstance(data, dict):
            raise ValueError("Search node state must be an object")
        allowed = {item.name for item in fields(SearchNode)}
        values = {key: value for key, value in data.items() if key in allowed}
        if "node_id" not in values:
            raise ValueError("Search node state is missing node_id")
        node_id = values["node_id"]
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0:
            raise ValueError("Search node node_id must be a non-negative integer")
        parent_node_id = values.get("parent_node_id")
        if parent_node_id is not None and (
            isinstance(parent_node_id, bool)
            or not isinstance(parent_node_id, int)
            or parent_node_id < 0
        ):
            raise ValueError("Search node parent_node_id must be null or non-negative")
        execution_revision = values.get("execution_revision", 0)
        if (
            isinstance(execution_revision, bool)
            or not isinstance(execution_revision, int)
            or execution_revision < 0
        ):
            raise ValueError("Search node execution_revision must be non-negative")

        string_fields = {
            "solution",
            "branch_name",
            "parent_branch_name",
            "implementation_base_ref",
            "diff_base_ref",
            "feedback_base_ref",
            "code_changes_summary",
            "agent_output",
            "evaluation_script_path",
            "evaluation_output",
            "feedback",
            "error_message",
            "workspace_dir",
            "code_diff",
            "technical_difficulties",
            "external_evaluation_error",
            "evaluation_integrity_error",
            "started_at",
        }
        invalid_strings = sorted(
            name
            for name in string_fields
            if name in values and not isinstance(values[name], str)
        )
        if invalid_strings:
            raise ValueError(
                "Search node string fields are invalid: " + ", ".join(invalid_strings)
            )

        for name in ("idea_id", "selection_batch_id"):
            value = values.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Search node {name} must be non-empty or null")

        for name in (
            "should_stop",
            "evaluation_valid",
            "had_error",
            "recoverable_error",
        ):
            if name in values and not isinstance(values[name], bool):
                raise ValueError(f"Search node {name} must be a boolean")

        provenance = values.get("evaluation_provenance", AGENT_GENERATED)
        if not isinstance(provenance, str) or provenance not in VALID_PROVENANCE:
            raise ValueError(
                "Search node evaluation_provenance must be 'provided' or "
                "'agent_generated'"
            )

        score = values.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("Search node score must be finite or null")

        for name in ("duration_seconds", "cost_usd"):
            value = values.get(name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(
                    f"Search node {name} must be finite and non-negative or null"
                )

        phase_telemetry = values.get("phase_telemetry", {})
        if not isinstance(phase_telemetry, dict):
            raise ValueError("Search node phase_telemetry must be an object")
        for phase_name, phase_values in phase_telemetry.items():
            if not isinstance(phase_name, str) or not isinstance(phase_values, dict):
                raise ValueError(
                    "Search node phase_telemetry must map phase names to objects"
                )
            for metric_name, metric_value in phase_values.items():
                if (
                    not isinstance(metric_name, str)
                    or isinstance(metric_value, bool)
                    or not isinstance(metric_value, (int, float))
                    or not math.isfinite(float(metric_value))
                    or float(metric_value) < 0
                ):
                    raise ValueError(
                        "Search node phase_telemetry values must be finite "
                        "and non-negative"
                    )

        for name in ("build_fidelity", "eval_fidelity"):
            if name in values and values[name] not in FIDELITIES:
                raise ValueError(
                    f"Search node {name} must be one of {sorted(FIDELITIES)}"
                )
        promoted_from = values.get("promoted_from")
        if promoted_from is not None and (
            isinstance(promoted_from, bool)
            or not isinstance(promoted_from, int)
            or promoted_from < 0
        ):
            raise ValueError("Search node promoted_from must be null or non-negative")
        raw_attempts = values.get("evaluation_attempts", [])
        if not isinstance(raw_attempts, list):
            raise ValueError("Search node evaluation_attempts must be a list")
        values["evaluation_attempts"] = [
            (
                attempt
                if isinstance(attempt, EvaluationAttempt)
                else EvaluationAttempt.from_dict(attempt)
            )
            for attempt in raw_attempts
        ]

        metrics, primary_metric = normalize_metrics(
            values.get("metrics", {}),
            values.get("primary_metric"),
        )
        values["metrics"] = metrics
        values["primary_metric"] = primary_metric
        values["external_evaluation_metadata"] = normalize_metadata(
            values.get("external_evaluation_metadata", {})
        )
        return cls(**values)

    def __str__(self) -> str:
        if self.had_error:
            return (
                f"- Node {self.node_id} failed: {self.error_message[:100]}...\n"
                f"  Solution: {self.solution[:200]}..."
            )
        return (
            f"- Node {self.node_id} (score={self.score}):\n"
            f"  Solution: {self.solution[:200]}...\n"
            + (f"  Feedback: {self.feedback[:200]}...\n" if self.feedback else "")
        )


__all__ = ["SearchNode"]
