"""Append-only, atomically written evaluator version registry.

The registry is the evaluator_id versioning backbone: one entry per
registered evaluator version, each carrying provenance, fidelity support,
and the measured timing model. A new ``evaluator_id`` can only come from a
maintainer registration — any other change to the evaluation tree is an
integrity violation on the candidate that made it.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


VALID_PROVENANCES = {"provided", "maintainer_built", "derived_fast"}


class EvaluationRegistryError(RuntimeError):
    """Raised for structurally invalid or inconsistent registry state."""


def _require_finite_non_negative(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise EvaluationRegistryError(
            f"{name} must be finite and non-negative"
        )
    return float(value)


@dataclass(frozen=True)
class TimingModel:
    """Measured evaluation timing: per-item rate plus fixed startup.

    Bootstrapped by the maintainer's calibration slice; refined by
    ``record_run`` samples. ``upper_seconds`` estimates use the worst
    observed sample times an overhead factor, so promises never ride on
    an average.
    """

    per_item_seconds: float
    startup_seconds: float
    total_items: int
    measured_samples: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_finite_non_negative("per_item_seconds", self.per_item_seconds)
        _require_finite_non_negative("startup_seconds", self.startup_seconds)
        if (
            isinstance(self.total_items, bool)
            or not isinstance(self.total_items, int)
            or self.total_items <= 0
        ):
            raise EvaluationRegistryError(
                "total_items must be a positive integer"
            )

    def expected_seconds(self, fraction: float) -> float:
        items = max(1, round(self.total_items * fraction))
        estimate = self.startup_seconds + items * self.per_item_seconds
        if self.measured_samples:
            same_fraction = [
                sample["duration_seconds"]
                for sample in self.measured_samples
                if abs(sample["fraction"] - fraction) < 1e-9
            ]
            if same_fraction:
                return sum(same_fraction) / len(same_fraction)
        return estimate

    def upper_seconds(self, fraction: float, overhead_factor: float) -> float:
        base = self.expected_seconds(fraction)
        worst = max(
            (
                sample["duration_seconds"]
                for sample in self.measured_samples
                if abs(sample["fraction"] - fraction) < 1e-9
            ),
            default=base,
        )
        return max(base, worst) * overhead_factor

    def with_sample(self, fraction: float, duration_seconds: float) -> "TimingModel":
        sample = {
            "fraction": _require_finite_non_negative("fraction", fraction),
            "duration_seconds": _require_finite_non_negative(
                "duration_seconds", duration_seconds
            ),
        }
        return TimingModel(
            per_item_seconds=self.per_item_seconds,
            startup_seconds=self.startup_seconds,
            total_items=self.total_items,
            measured_samples=[*self.measured_samples, sample],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimingModel":
        if not isinstance(data, Mapping):
            raise EvaluationRegistryError("Timing model must be an object")
        return cls(
            per_item_seconds=data.get("per_item_seconds", 0.0),
            startup_seconds=data.get("startup_seconds", 0.0),
            total_items=data.get("total_items", 0),
            measured_samples=list(data.get("measured_samples", [])),
        )


@dataclass(frozen=True)
class EvaluatorVersion:
    """One registered evaluator version."""

    evaluator_id: str  # manifest fingerprint of the evaluation tree
    version: int
    provenance: str  # provided | maintainer_built | derived_fast
    parent_evaluator: Optional[str]
    fidelity_support: Dict[str, Any]  # e.g. {"fast_fraction": 0.15, "seed": 1337}
    timing: TimingModel
    created_at_iteration: int
    reason: str
    # The inputs half of evaluation identity: hashes of the protected
    # data files the evaluation reads. Empty when no paths are protected.
    data_manifest: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator_id, str) or not self.evaluator_id:
            raise EvaluationRegistryError(
                "evaluator_id must be a non-empty string"
            )
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise EvaluationRegistryError("version must be a positive integer")
        if self.provenance not in VALID_PROVENANCES:
            raise EvaluationRegistryError(
                f"provenance must be one of {sorted(VALID_PROVENANCES)}"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise EvaluationRegistryError("reason must be a non-empty string")
        if not isinstance(self.data_manifest, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.data_manifest.items()
        ):
            raise EvaluationRegistryError(
                "data_manifest must map file paths to digests"
            )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timing"] = self.timing.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluatorVersion":
        if not isinstance(data, Mapping):
            raise EvaluationRegistryError(
                "Evaluator version must be an object"
            )
        values = dict(data)
        values["timing"] = TimingModel.from_dict(values.get("timing", {}))
        return cls(**values)


class EvaluationRegistry:
    """Reader/writer for ``.kapso/evaluation_registry.json`` (append-only)."""

    RELATIVE_PATH = Path(".kapso") / "evaluation_registry.json"

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.path = self.workspace_dir / self.RELATIVE_PATH

    def exists(self) -> bool:
        return self.path.is_file()

    def versions(self) -> List[EvaluatorVersion]:
        if not self.path.is_file():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise EvaluationRegistryError(
                f"Evaluation registry must be a JSON list: {self.path}"
            )
        return [EvaluatorVersion.from_dict(item) for item in raw]

    def head(self) -> Optional[EvaluatorVersion]:
        versions = self.versions()
        return versions[-1] if versions else None

    def register(self, version: EvaluatorVersion) -> None:
        """Append one version. Never mutates or removes prior entries."""
        existing = self.versions()
        if existing:
            expected = existing[-1].version + 1
            if version.version != expected:
                raise EvaluationRegistryError(
                    f"Expected version {expected}, got {version.version}"
                )
            if any(v.evaluator_id == version.evaluator_id for v in existing):
                raise EvaluationRegistryError(
                    "evaluator_id already registered: "
                    f"{version.evaluator_id}"
                )
        elif version.version != 1:
            raise EvaluationRegistryError(
                f"First registered version must be 1, got {version.version}"
            )
        self._write([*existing, version])

    def update_head_timing(self, timing: TimingModel) -> None:
        """Replace the head's timing model (measurement refinement only)."""
        versions = self.versions()
        if not versions:
            raise EvaluationRegistryError(
                "Cannot update timing: registry is empty"
            )
        head = versions[-1]
        refined = EvaluatorVersion(
            evaluator_id=head.evaluator_id,
            version=head.version,
            provenance=head.provenance,
            parent_evaluator=head.parent_evaluator,
            fidelity_support=head.fidelity_support,
            timing=timing,
            created_at_iteration=head.created_at_iteration,
            reason=head.reason,
            data_manifest=head.data_manifest,
        )
        self._write([*versions[:-1], refined])

    def _write(self, versions: List[EvaluatorVersion]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            [version.to_dict() for version in versions],
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        descriptor, temp_path = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=".evaluation_registry.",
            suffix=".tmp",
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.path)
