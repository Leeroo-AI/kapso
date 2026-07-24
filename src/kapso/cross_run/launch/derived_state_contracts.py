"""Content-addressed contracts for checkpoint-governed derived run state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import ClassVar, Mapping

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import ContractValidationError, StrictContract
from kapso.cross_run.launch.contracts import BootstrapPin

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_CHECKPOINT_NAMESPACE = "run-checkpoint"
_CHECKPOINT_HEAD_NAMESPACE = "run-checkpoint-head"
_EVIDENCE_NAMESPACE = "run-derivative-evidence"
_STRATEGY_AUTHORITIES = {
    "generic": frozenset(
        {
            "idea_archive",
            "experiment_history",
            "execution_journal",
            "action_ledger",
        }
    ),
    "benchmark_tree_search": frozenset(
        {
            "experiment_history",
            "execution_journal",
            "action_ledger",
        }
    ),
}


class DerivedStateContractError(ContractValidationError):
    """A derived-state layout or payload generation is not exact."""


class RunStateAuthority(str, Enum):
    """Mutable run-state projections governed by the durable checkpoint."""

    IDEA_ARCHIVE = "idea_archive"
    EXPERIMENT_HISTORY = "experiment_history"
    EXECUTION_JOURNAL = "execution_journal"
    ACTION_LEDGER = "action_ledger"


class RunStatePayloadFormat(str, Enum):
    """Canonical byte encodings supported by derived-state promotion."""

    CANONICAL_JSON = "canonical_json"
    CANONICAL_JSONL = "canonical_jsonl"


_AUTHORITY_FORMATS = {
    RunStateAuthority.IDEA_ARCHIVE: RunStatePayloadFormat.CANONICAL_JSON,
    RunStateAuthority.EXPERIMENT_HISTORY: RunStatePayloadFormat.CANONICAL_JSON,
    RunStateAuthority.EXECUTION_JOURNAL: RunStatePayloadFormat.CANONICAL_JSONL,
    RunStateAuthority.ACTION_LEDGER: RunStatePayloadFormat.CANONICAL_JSON,
}


def _require_namespace(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise DerivedStateContractError(f"{name} uses the wrong namespace")


def _require_digest(value: str, name: str) -> None:
    if _DIGEST_PATTERN.fullmatch(value) is None:
        raise DerivedStateContractError(f"{name} must be a sha256 digest")


def _require_relative_path(value: str, name: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise DerivedStateContractError(
            f"{name} must be a normalized non-dot relative path"
        )
    return path


def _paths_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left in right.parents or right in left.parents


@dataclass(frozen=True)
class RunStateAuthorityBinding(StrictContract):
    """One exact mutable authority path and its canonical payload encoding."""

    authority_binding_id: str
    authority: RunStateAuthority
    relative_path: str
    payload_format: RunStatePayloadFormat

    CONTENT_NAMESPACE: ClassVar[str] = "run-state-authority-binding"
    IDENTITY_FIELD: ClassVar[str] = "authority_binding_id"

    def _validate(self) -> None:
        _require_relative_path(self.relative_path, "run-state authority path")
        if self.payload_format is not _AUTHORITY_FORMATS[self.authority]:
            raise DerivedStateContractError(
                "run-state authority uses the wrong payload format"
            )

    @classmethod
    def build(
        cls,
        *,
        authority: RunStateAuthority,
        relative_path: str,
    ) -> "RunStateAuthorityBinding":
        if type(authority) is not RunStateAuthority:
            raise DerivedStateContractError(
                "run-state authority binding requires one typed authority"
            )
        return cls.mint(
            authority=authority,
            relative_path=relative_path,
            payload_format=_AUTHORITY_FORMATS[authority],
        )


@dataclass(frozen=True)
class RunStateLayout(StrictContract):
    """The complete, non-overlapping mutable-state layout for one strategy."""

    layout_id: str
    strategy_kind: str
    bindings: tuple[RunStateAuthorityBinding, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-state-layout"
    IDENTITY_FIELD: ClassVar[str] = "layout_id"

    def _validate(self) -> None:
        required_authorities = _STRATEGY_AUTHORITIES.get(self.strategy_kind)
        if required_authorities is None:
            raise DerivedStateContractError(
                "run-state layout strategy kind is unsupported"
            )
        if any(
            type(binding) is not RunStateAuthorityBinding for binding in self.bindings
        ):
            raise DerivedStateContractError(
                "run-state layout requires exact authority bindings"
            )
        authority_values = tuple(binding.authority.value for binding in self.bindings)
        if authority_values != tuple(sorted(authority_values)):
            raise DerivedStateContractError(
                "run-state authority bindings must be in authority order"
            )
        if set(authority_values) != required_authorities:
            raise DerivedStateContractError(
                "run-state layout authorities are incomplete"
            )
        paths = tuple(
            _require_relative_path(binding.relative_path, "run-state authority path")
            for binding in self.bindings
        )
        for position, path in enumerate(paths):
            if any(_paths_overlap(path, other) for other in paths[position + 1 :]):
                raise DerivedStateContractError("run-state authority paths overlap")

    @classmethod
    def build(
        cls,
        *,
        strategy_kind: str,
        authority_paths: Mapping[RunStateAuthority, str],
    ) -> "RunStateLayout":
        if not isinstance(authority_paths, Mapping) or any(
            type(authority) is not RunStateAuthority for authority in authority_paths
        ):
            raise DerivedStateContractError(
                "run-state layout requires an explicit typed authority-path mapping"
            )
        bindings = tuple(
            RunStateAuthorityBinding.build(
                authority=authority,
                relative_path=authority_paths[authority],
            )
            for authority in sorted(authority_paths, key=lambda item: item.value)
        )
        return cls.mint(
            strategy_kind=strategy_kind,
            bindings=bindings,
        )


@dataclass(frozen=True)
class RunStatePayloadTransition(StrictContract):
    """One staged authority payload and its exact revision transition."""

    payload_transition_id: str
    authority_binding_id: str
    predecessor_digest: str | None
    predecessor_revision: int | None
    predecessor_size_bytes: int | None
    target_digest: str
    target_revision: int
    target_size_bytes: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-state-payload-transition"
    IDENTITY_FIELD: ClassVar[str] = "payload_transition_id"

    def _validate(self) -> None:
        _require_namespace(
            self.authority_binding_id,
            RunStateAuthorityBinding.CONTENT_NAMESPACE,
            "run-state payload authority binding",
        )
        absent_predecessor_fields = (
            self.predecessor_digest is None,
            self.predecessor_revision is None,
            self.predecessor_size_bytes is None,
        )
        if len(set(absent_predecessor_fields)) != 1:
            raise DerivedStateContractError(
                "run-state payload predecessor fields differ"
            )
        _require_digest(self.target_digest, "run-state payload target digest")
        if type(self.target_revision) is not int or self.target_revision < 0:
            raise DerivedStateContractError(
                "run-state payload target revision must be non-negative"
            )
        if type(self.target_size_bytes) is not int or self.target_size_bytes < 0:
            raise DerivedStateContractError(
                "run-state payload target size must be non-negative"
            )
        if self.predecessor_digest is None:
            if self.target_revision != 0:
                raise DerivedStateContractError(
                    "genesis run-state payload must target revision zero"
                )
            return
        _require_digest(
            self.predecessor_digest,
            "run-state payload predecessor digest",
        )
        if self.predecessor_revision < 0:
            raise DerivedStateContractError(
                "run-state payload predecessor revision must be non-negative"
            )
        if self.predecessor_size_bytes < 0:
            raise DerivedStateContractError(
                "run-state payload predecessor size must be non-negative"
            )
        same_digest = self.target_digest == self.predecessor_digest
        same_revision = self.target_revision == self.predecessor_revision
        if self.target_revision < self.predecessor_revision:
            raise DerivedStateContractError(
                "run-state payload target revision cannot roll back"
            )
        if same_digest != same_revision:
            raise DerivedStateContractError(
                "run-state payload revision changes exactly when its digest changes"
            )
        if same_digest and self.target_size_bytes != self.predecessor_size_bytes:
            raise DerivedStateContractError(
                "unchanged run-state payload must retain its exact size"
            )


@dataclass(frozen=True)
class RunDerivedStateGeneration(StrictContract):
    """A complete staged projection generation named by checkpoint evidence."""

    generation_id: str
    bootstrap_pin_id: str
    run_state_layout: RunStateLayout
    predecessor_checkpoint_head_id: str
    predecessor_checkpoint_id: str | None
    predecessor_evidence_id: str | None
    target_evidence_id: str
    payload_transitions: tuple[RunStatePayloadTransition, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-derived-state-generation"
    IDENTITY_FIELD: ClassVar[str] = "generation_id"

    def _validate(self) -> None:
        _require_namespace(
            self.bootstrap_pin_id,
            BootstrapPin.CONTENT_NAMESPACE,
            "run derived-state bootstrap pin",
        )
        if type(self.run_state_layout) is not RunStateLayout:
            raise DerivedStateContractError(
                "run derived-state generation requires one exact layout"
            )
        _require_namespace(
            self.predecessor_checkpoint_head_id,
            _CHECKPOINT_HEAD_NAMESPACE,
            "run derived-state predecessor checkpoint head",
        )
        has_predecessor = self.predecessor_checkpoint_id is not None
        if has_predecessor != (self.predecessor_evidence_id is not None):
            raise DerivedStateContractError(
                "run derived-state predecessor fields differ"
            )
        if has_predecessor:
            _require_namespace(
                self.predecessor_checkpoint_id,
                _CHECKPOINT_NAMESPACE,
                "run derived-state predecessor checkpoint",
            )
            _require_namespace(
                self.predecessor_evidence_id,
                _EVIDENCE_NAMESPACE,
                "run derived-state predecessor evidence",
            )
        _require_namespace(
            self.target_evidence_id,
            _EVIDENCE_NAMESPACE,
            "run derived-state target evidence",
        )
        expected_binding_ids = tuple(
            binding.authority_binding_id for binding in self.run_state_layout.bindings
        )
        transition_binding_ids = tuple(
            transition.authority_binding_id for transition in self.payload_transitions
        )
        if (
            any(
                type(transition) is not RunStatePayloadTransition
                for transition in self.payload_transitions
            )
            or transition_binding_ids != expected_binding_ids
        ):
            raise DerivedStateContractError(
                "run derived-state payload transitions are incomplete or unordered"
            )
        if any(
            (
                transition.predecessor_digest is not None
                and transition.predecessor_revision is not None
                and transition.predecessor_size_bytes is not None
            )
            != has_predecessor
            for transition in self.payload_transitions
        ):
            raise DerivedStateContractError(
                "run derived-state payload frontier differs from its predecessor"
            )
        if self.exact_dependency_ids != tuple(sorted(set(self.exact_dependency_ids))):
            raise DerivedStateContractError(
                "run derived-state exact dependencies must be sorted and unique"
            )
        for dependency_id in self.exact_dependency_ids:
            require_content_id(
                dependency_id,
                "run derived-state exact dependency",
            )
        required_dependencies = {
            self.bootstrap_pin_id,
            self.run_state_layout.layout_id,
            self.predecessor_checkpoint_head_id,
            self.target_evidence_id,
            *(
                transition.payload_transition_id
                for transition in self.payload_transitions
            ),
        }
        if has_predecessor:
            required_dependencies.update(
                {
                    self.predecessor_checkpoint_id,
                    self.predecessor_evidence_id,
                }
            )
        if set(self.exact_dependency_ids) != required_dependencies:
            raise DerivedStateContractError(
                "run derived-state dependency closure is not exact"
            )

    @classmethod
    def build(
        cls,
        *,
        bootstrap_pin_id: str,
        run_state_layout: RunStateLayout,
        predecessor_checkpoint_head_id: str,
        predecessor_checkpoint_id: str | None,
        predecessor_evidence_id: str | None,
        target_evidence_id: str,
        payload_transitions: tuple[RunStatePayloadTransition, ...],
    ) -> "RunDerivedStateGeneration":
        dependencies = {
            bootstrap_pin_id,
            run_state_layout.layout_id,
            predecessor_checkpoint_head_id,
            target_evidence_id,
            *(transition.payload_transition_id for transition in payload_transitions),
        }
        if predecessor_checkpoint_id is not None:
            dependencies.add(predecessor_checkpoint_id)
        if predecessor_evidence_id is not None:
            dependencies.add(predecessor_evidence_id)
        return cls.mint(
            bootstrap_pin_id=bootstrap_pin_id,
            run_state_layout=run_state_layout,
            predecessor_checkpoint_head_id=predecessor_checkpoint_head_id,
            predecessor_checkpoint_id=predecessor_checkpoint_id,
            predecessor_evidence_id=predecessor_evidence_id,
            target_evidence_id=target_evidence_id,
            payload_transitions=payload_transitions,
            exact_dependency_ids=tuple(sorted(dependencies)),
        )


__all__ = [
    "DerivedStateContractError",
    "RunDerivedStateGeneration",
    "RunStateAuthority",
    "RunStateAuthorityBinding",
    "RunStateLayout",
    "RunStatePayloadFormat",
    "RunStatePayloadTransition",
]
