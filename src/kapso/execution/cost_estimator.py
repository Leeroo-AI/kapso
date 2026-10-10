"""Platform cost estimates, policy validation, and candidate selection utility."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

from kapso.core.config import load_platform_defaults
from kapso.execution.iteration_evaluator import IterationEvaluationContext


DEFAULT_COST_CONFIG = load_platform_defaults()["search_strategy"]["params"]["cost"]


@dataclass(frozen=True)
class CostEstimate:
    """A caller-owned estimate in the campaign's configured platform unit."""

    amount: float
    unit: str
    basis: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


CostEstimator = Callable[[IterationEvaluationContext], CostEstimate]


def _validate_amount(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite, numeric, and non-negative")


def normalize_cost_config(config: Optional[Mapping[str, Any]] = None) -> dict:
    """Resolve the policy from its single config source and reject bad inputs."""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("cost must be a mapping")
    values = dict(DEFAULT_COST_CONFIG)
    if config is not None:
        unknown = set(config) - values.keys()
        if unknown:
            raise ValueError(f"Unknown cost config keys: {sorted(unknown)}")
        values.update(config)
    if not isinstance(values["enabled"], bool):
        raise ValueError("cost.enabled must be a boolean")
    estimator = values["estimator"]
    if estimator is not None and (
        not isinstance(estimator, str)
        or "." not in estimator
        or not all(part.isidentifier() for part in estimator.split("."))
    ):
        raise ValueError("cost.estimator must be a module-qualified callable name")
    if values["enabled"] and estimator is None:
        raise ValueError("cost.estimator is required when cost.enabled is true")
    for name in ("cap_per_candidate", "campaign_budget"):
        if values[name] is not None:
            _validate_amount(values[name], f"cost.{name}")
    _validate_amount(values["weight"], "cost.weight")
    if not isinstance(values["unit"], str) or not values["unit"].strip():
        raise ValueError("cost.unit must be a non-empty string")
    if values["history_field"] not in ("estimated_cost", "measured_cost"):
        raise ValueError("cost.history_field must be estimated_cost or measured_cost")
    return values


def load_cost_estimator(config: Mapping[str, Any]) -> Optional[CostEstimator]:
    """Load a configured estimator; importing or resolving it must succeed."""
    policy = normalize_cost_config(config)
    if not policy["enabled"]:
        return None
    module_name, _, attribute = policy["estimator"].rpartition(".")
    estimator = getattr(importlib.import_module(module_name), attribute)
    if not callable(estimator):
        raise ValueError(f"cost.estimator {policy['estimator']!r} must be callable")
    return estimator


def _validate_metadata_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Cost estimate metadata keys must be strings")
        for nested in value.values():
            _validate_metadata_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_metadata_keys(nested)


def validate_cost_value(
    value: Mapping[str, Any],
    expected_unit: Optional[str] = None,
    *,
    estimated: bool = False,
) -> dict:
    """Validate a stored measurement or estimate and detach its metadata."""
    if not isinstance(value, Mapping):
        raise ValueError("Candidate cost must be an object")
    amount = value.get("amount")
    _validate_amount(amount, "Candidate cost amount")
    unit = value.get("unit")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("Candidate cost unit must be a non-empty string")
    if expected_unit is not None and unit != expected_unit:
        raise ValueError(f"Candidate cost unit {unit!r} does not match cost.unit {expected_unit!r}")
    result = {"amount": float(amount), "unit": unit}
    if estimated:
        basis = value.get("basis")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("Cost estimate basis must be a non-empty string")
        metadata = value.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Cost estimate metadata must be a mapping")
        _validate_metadata_keys(metadata)
        result.update(basis=basis, metadata=json.loads(json.dumps(dict(metadata), allow_nan=False)))
    return result


def normalize_cost_estimate(result: CostEstimate, expected_unit: str) -> dict:
    """Require the estimator result contract and return durable node data."""
    if not isinstance(result, CostEstimate):
        raise ValueError("Cost estimator must return CostEstimate")
    return validate_cost_value(
        {"amount": result.amount, "unit": result.unit, "basis": result.basis, "metadata": result.metadata},
        expected_unit,
        estimated=True,
    )


def validate_cost_refusal(value: Mapping[str, Any]) -> dict:
    """Validate the durable reason why a finalized candidate did not run."""
    if not isinstance(value, Mapping):
        raise ValueError("Candidate cost_refusal must be an object")
    estimate = validate_cost_value(value.get("estimate"), estimated=True)
    cap = value.get("cap")
    _validate_amount(cap, "Candidate cost_refusal cap")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Candidate cost_refusal reason must be non-empty")
    result = {"estimate": estimate, "cap": float(cap), "reason": reason}
    if "campaign_cost_used" in value:
        _validate_amount(value["campaign_cost_used"], "Candidate cost_refusal campaign_cost_used")
        result["campaign_cost_used"] = float(value["campaign_cost_used"])
    return result


def effective_cost(node: Any, config: Mapping[str, Any]) -> float:
    """Measured platform cost supersedes an estimate, including a zero cost."""
    if not config["enabled"]:
        return 0.0
    value = getattr(node, "measured_cost", None)
    if value is None:
        value = getattr(node, "estimated_cost", None)
    if value is None:
        raise ValueError(f"Candidate {node.node_id} is missing platform cost")
    return validate_cost_value(value, config["unit"])["amount"]


def campaign_cost_used(nodes: Iterable[Any], config: Mapping[str, Any]) -> float:
    """Charge each evaluated candidate once; refused estimates spent nothing."""
    used = 0.0
    for node in nodes:
        if getattr(node, "cost_refusal", None):
            continue
        # A failed or suspended build may never have reached admission.
        if (
            (node.score is None or getattr(node, "suspended", False))
            and getattr(node, "estimated_cost", None) is None
            and getattr(node, "measured_cost", None) is None
        ):
            continue
        used += effective_cost(node, config)
    return used


def selection_score(node: Any, config: Mapping[str, Any]) -> float:
    """Higher is better: signed score minus weighted platform cost.

    ``config`` carries ``cost`` and ``maximize_scoring``. The internal score
    remains unchanged; minimizing problems negate it before the same penalty.
    """
    if node.score is None:
        return -math.inf
    score = node.score if config["maximize_scoring"] else -node.score
    policy = config["cost"]
    if policy is None:
        policy = DEFAULT_COST_CONFIG
    if not policy["enabled"]:
        return score
    return score - policy["weight"] * effective_cost(node, policy)
