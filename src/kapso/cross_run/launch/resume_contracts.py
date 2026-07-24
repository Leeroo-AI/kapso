"""Content-addressed derivative and safety authorities for resumed runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Mapping

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.launch.contracts import BootstrapPin
from kapso.cross_run.git_refs import require_git_ref_name
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ResumeContractError(ValueError):
    """A resume derivative or safety authority is invalid."""


class RunDerivativeKind(str, Enum):
    """Kinds of content-addressed evidence created inside one run."""

    IDEA = "idea"
    EXPERIMENT = "experiment"
    REVISION = "revision"
    ARTIFACT = "artifact"


class RunSafetyBoundary(str, Enum):
    """Dangerous boundaries requiring freshly checked run safety."""

    INITIALIZATION = "initialization"
    RESUME = "resume"
    IDEATION = "ideation"
    IMPLEMENTATION = "implementation"
    EVALUATION = "evaluation"
    PUBLICATION = "publication"


class RunReleaseUseMode(str, Enum):
    """How release-use policy was established for this boundary."""

    ONLINE_CURRENT = "online_current"
    PINNED_OFFLINE = "pinned_offline"


class RunEligibilityDisposition(str, Enum):
    """Derived permissions for one exact run frontier."""

    ELIGIBLE = "eligible"
    REPRODUCIBILITY_ONLY = "reproducibility_only"
    SECURITY_BLOCKED = "security_blocked"


def _require_sorted_content_ids(
    values: tuple[str, ...],
    name: str,
    *,
    required: bool = False,
) -> None:
    if required and not values:
        raise ResumeContractError(f"{name} must not be empty")
    if values != tuple(sorted(set(values))):
        raise ResumeContractError(f"{name} must be sorted and unique")
    for value in values:
        require_content_id(value, name)


def _release_use_subject_ids(
    observation: ExpertReleaseUsePolicyObservation,
) -> tuple[str, ...]:
    subjects = {
        observation.observation_id,
        observation.scope_contract_id,
        observation.knowledge_snapshot_id,
        observation.knowledge_publication_id,
        *observation.checked_release_ids,
    }
    for revocation in observation.matched_revocations:
        subjects.update(
            {
                revocation.revocation_id,
                revocation.release_id,
                revocation.release_publication_id,
                revocation.release_activation_witness_id,
                *revocation.exact_evidence_refs,
            }
        )
    return tuple(sorted(subjects))


def _release_use_authority_coordinates(
    observation: ExpertReleaseUsePolicyObservation,
) -> tuple[object, ...]:
    return (
        observation.scope_id,
        observation.scope_contract_id,
        observation.scope_repository_binding_hash,
        observation.repository_full_name,
        observation.repository_node_id,
        observation.knowledge_snapshot_id,
        observation.catalog_generation,
        observation.knowledge_publication_id,
        observation.current_pointer_digest,
        observation.authority_commit_sha,
        observation.release_attestation_ref,
    )


def _security_authority_coordinates(
    observation: SecurityDenylistObservation,
) -> tuple[object, ...]:
    return (
        observation.scope_id,
        observation.scope_contract_id,
        observation.scope_repository_binding_hash,
        observation.snapshot_id,
        observation.generation,
        observation.publication_id,
        observation.repository_full_name,
        observation.repository_node_id,
        observation.pointer_digest,
        observation.authority_commit_sha,
        observation.release_attestation_ref,
    )


@dataclass(frozen=True)
class RunDerivativeRecord(StrictContract):
    """One immutable, locally addressable derivative of launch evidence."""

    derivative_id: str
    kind: RunDerivativeKind
    local_locator: str
    payload_digest: str
    authorization_safety_state_id: str
    direct_source_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-derivative"
    IDENTITY_FIELD: ClassVar[str] = "derivative_id"

    def _validate(self) -> None:
        if (
            not isinstance(self.local_locator, str)
            or not self.local_locator.strip()
            or "\x00" in self.local_locator
        ):
            raise ResumeContractError(
                "run derivative local locator must be non-empty text"
            )
        if _DIGEST_PATTERN.fullmatch(self.payload_digest) is None:
            raise ResumeContractError("run derivative payload digest must be sha256")
        require_content_id(
            self.authorization_safety_state_id,
            "run derivative authorization safety state",
        )
        _require_sorted_content_ids(
            self.direct_source_ids,
            "run derivative direct sources",
            required=True,
        )
        if self.authorization_safety_state_id not in self.direct_source_ids:
            raise ResumeContractError(
                "run derivative omits its authorizing safety state"
            )
        if self.derivative_id in self.direct_source_ids:
            raise ResumeContractError("run derivative cannot directly source itself")


@dataclass(frozen=True)
class RunBranchAdvance(StrictContract):
    """One authorized, append-only transition of a run workspace branch."""

    branch_advance_id: str
    branch: str
    predecessor_commit_sha: str
    commit_sha: str
    predecessor_branch_advance_id: str | None
    authorization_safety_state_id: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-branch-advance"
    IDENTITY_FIELD: ClassVar[str] = "branch_advance_id"

    def _validate(self) -> None:
        require_git_ref_name(
            f"refs/heads/{self.branch}",
            "run branch advance branch",
            qualified=True,
            error_type=ResumeContractError,
        )
        for commit_sha, name in (
            (self.predecessor_commit_sha, "predecessor"),
            (self.commit_sha, "successor"),
        ):
            if re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None:
                raise ResumeContractError(
                    f"run branch advance {name} must be a Git object ID"
                )
        if self.predecessor_commit_sha == self.commit_sha:
            raise ResumeContractError("run branch advance must change the branch head")
        if self.predecessor_branch_advance_id is not None:
            require_content_id(
                self.predecessor_branch_advance_id,
                "run branch advance predecessor",
            )
            if (
                self.predecessor_branch_advance_id.split(":sha256:", 1)[0]
                != self.CONTENT_NAMESPACE
            ):
                raise ResumeContractError(
                    "run branch advance predecessor uses the wrong namespace"
                )
        require_content_id(
            self.authorization_safety_state_id,
            "run branch advance authorization safety state",
        )
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "run branch advance exact dependencies",
            required=True,
        )
        expected_dependencies = {self.authorization_safety_state_id}
        if self.predecessor_branch_advance_id is not None:
            expected_dependencies.add(self.predecessor_branch_advance_id)
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ResumeContractError(
                "run branch advance dependency closure is not exact"
            )

    @classmethod
    def build(
        cls,
        *,
        branch: str,
        predecessor_commit_sha: str,
        commit_sha: str,
        predecessor_branch_advance_id: str | None,
        authorization_safety_state_id: str,
    ) -> "RunBranchAdvance":
        dependencies = {authorization_safety_state_id}
        if predecessor_branch_advance_id is not None:
            dependencies.add(predecessor_branch_advance_id)
        return cls.mint(
            branch=branch,
            predecessor_commit_sha=predecessor_commit_sha,
            commit_sha=commit_sha,
            predecessor_branch_advance_id=predecessor_branch_advance_id,
            authorization_safety_state_id=authorization_safety_state_id,
            exact_dependency_ids=tuple(sorted(dependencies)),
        )


@dataclass(frozen=True)
class RunDerivativeEvidence(StrictContract):
    """Exact mutable-store, Git, and artifact frontier reconciled from disk."""

    evidence_id: str
    state_authority_digests: Mapping[str, str]
    state_authority_revisions: Mapping[str, int]
    branch_origin_heads: Mapping[str, str]
    branch_advances: tuple[RunBranchAdvance, ...]
    branch_heads: Mapping[str, str]
    artifact_digests: Mapping[str, str]
    derivative_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-derivative-evidence"
    IDENTITY_FIELD: ClassVar[str] = "evidence_id"

    def _validate(self) -> None:
        if not self.state_authority_digests or set(self.state_authority_digests) != set(
            self.state_authority_revisions
        ):
            raise ResumeContractError(
                "run derivative evidence state authorities are incomplete"
            )
        for authority, digest in self.state_authority_digests.items():
            require_identifier(authority, "run derivative state authority")
            if _DIGEST_PATTERN.fullmatch(digest) is None:
                raise ResumeContractError(
                    "run derivative state authority digest must be sha256"
                )
            revision = self.state_authority_revisions[authority]
            if type(revision) is not int or revision < 0:
                raise ResumeContractError(
                    "run derivative state authority revision must be non-negative"
                )
        if not self.branch_origin_heads or set(self.branch_origin_heads) != set(
            self.branch_heads
        ):
            raise ResumeContractError(
                "run derivative branch origins and heads are incomplete"
            )
        for branch, commit_sha in {
            **self.branch_origin_heads,
            **self.branch_heads,
        }.items():
            require_git_ref_name(
                f"refs/heads/{branch}",
                "run derivative branch",
                qualified=True,
                error_type=ResumeContractError,
            )
            if re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None:
                raise ResumeContractError(
                    "run derivative branch head must be a Git object ID"
                )
        advance_ids = tuple(item.branch_advance_id for item in self.branch_advances)
        if advance_ids != tuple(sorted(set(advance_ids))):
            raise ResumeContractError(
                "run derivative branch advances must be sorted and unique"
            )
        advances_by_branch: dict[str, list[RunBranchAdvance]] = {
            branch: [] for branch in self.branch_heads
        }
        for advance in self.branch_advances:
            if (
                type(advance) is not RunBranchAdvance
                or advance.branch not in advances_by_branch
            ):
                raise ResumeContractError(
                    "run derivative branch advance has an unknown branch"
                )
            advances_by_branch[advance.branch].append(advance)
        for branch, advances in advances_by_branch.items():
            remaining = {advance.branch_advance_id: advance for advance in advances}
            predecessor_advance_id: str | None = None
            commit_sha = self.branch_origin_heads[branch]
            while remaining:
                candidates = tuple(
                    advance
                    for advance in remaining.values()
                    if (
                        advance.predecessor_branch_advance_id == predecessor_advance_id
                        and advance.predecessor_commit_sha == commit_sha
                    )
                )
                if len(candidates) != 1:
                    raise ResumeContractError(
                        "run derivative branch advances are not one exact chain"
                    )
                advance = candidates[0]
                del remaining[advance.branch_advance_id]
                predecessor_advance_id = advance.branch_advance_id
                commit_sha = advance.commit_sha
            if commit_sha != self.branch_heads[branch]:
                raise ResumeContractError(
                    "run derivative branch head differs from its advance chain"
                )
        for locator, digest in self.artifact_digests.items():
            if (
                not isinstance(locator, str)
                or not locator.strip()
                or "\x00" in locator
                or _DIGEST_PATTERN.fullmatch(digest) is None
            ):
                raise ResumeContractError("run derivative artifact evidence is invalid")
        _require_sorted_content_ids(
            self.derivative_ids,
            "run derivative evidence IDs",
        )


@dataclass(frozen=True)
class RunDerivativeFrontier(StrictContract):
    """Exact acyclic derivative DAG and its current terminal evidence."""

    frontier_id: str
    launch_subject_ids: tuple[str, ...]
    evidence: RunDerivativeEvidence
    derivatives: tuple[RunDerivativeRecord, ...]
    terminal_derivative_ids: tuple[str, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-derivative-frontier"
    IDENTITY_FIELD: ClassVar[str] = "frontier_id"

    def _validate(self) -> None:
        _require_sorted_content_ids(
            self.launch_subject_ids,
            "run frontier launch subjects",
            required=True,
        )
        if type(self.evidence) is not RunDerivativeEvidence:
            raise ResumeContractError(
                "run frontier requires exact reconciled derivative evidence"
            )
        derivative_ids = tuple(item.derivative_id for item in self.derivatives)
        if derivative_ids != tuple(sorted(set(derivative_ids))):
            raise ResumeContractError(
                "run frontier derivatives must be sorted and unique"
            )
        local_identities = tuple(
            (item.kind, item.local_locator) for item in self.derivatives
        )
        if len(local_identities) != len(set(local_identities)):
            raise ResumeContractError("run frontier derivatives reuse a local identity")
        derivative_id_set = set(derivative_ids)
        authorization_ids = {
            derivative.authorization_safety_state_id for derivative in self.derivatives
        }
        branch_advance_ids = {
            advance.branch_advance_id for advance in self.evidence.branch_advances
        }
        branch_advance_dependency_ids = {
            dependency_id
            for advance in self.evidence.branch_advances
            for dependency_id in advance.exact_dependency_ids
        }
        known = (
            set(self.launch_subject_ids)
            | derivative_id_set
            | authorization_ids
            | branch_advance_ids
            | branch_advance_dependency_ids
            | {self.evidence.evidence_id}
        )
        graph = {
            derivative.derivative_id: tuple(
                source_id
                for source_id in derivative.direct_source_ids
                if source_id in derivative_id_set
            )
            for derivative in self.derivatives
        }
        for derivative in self.derivatives:
            if not set(derivative.direct_source_ids).issubset(known):
                raise ResumeContractError(
                    "run frontier derivative references an unknown source"
                )
        visited: set[str] = set()
        active: set[str] = set()

        def visit(derivative_id: str) -> None:
            if derivative_id in active:
                raise ResumeContractError("run frontier derivative graph has a cycle")
            if derivative_id in visited:
                return
            active.add(derivative_id)
            for source_id in graph[derivative_id]:
                visit(source_id)
            active.remove(derivative_id)
            visited.add(derivative_id)

        for derivative_id in derivative_ids:
            visit(derivative_id)
        sourced_derivatives = {
            source_id
            for derivative in self.derivatives
            for source_id in derivative.direct_source_ids
            if source_id in derivative_id_set
        }
        expected_terminals = tuple(sorted(derivative_id_set - sourced_derivatives))
        if self.terminal_derivative_ids != expected_terminals:
            raise ResumeContractError(
                "run frontier terminal derivatives are not the exact DAG leaves"
            )
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "run frontier exact dependencies",
            required=True,
        )
        if set(self.exact_dependency_ids) != known:
            raise ResumeContractError("run frontier dependency closure is not exact")
        if self.evidence.derivative_ids != derivative_ids:
            raise ResumeContractError(
                "run frontier derivatives differ from reconciled evidence"
            )
        expected_artifacts = {
            derivative.local_locator: derivative.payload_digest
            for derivative in self.derivatives
            if derivative.kind is RunDerivativeKind.ARTIFACT
        }
        if dict(self.evidence.artifact_digests) != expected_artifacts:
            raise ResumeContractError(
                "run frontier artifacts differ from reconciled evidence"
            )

    @classmethod
    def build(
        cls,
        *,
        launch_subject_ids: tuple[str, ...],
        evidence: RunDerivativeEvidence,
        derivatives: tuple[RunDerivativeRecord, ...],
    ) -> "RunDerivativeFrontier":
        ordered = tuple(sorted(derivatives, key=lambda item: item.derivative_id))
        derivative_ids = {item.derivative_id for item in ordered}
        branch_advance_ids = {
            advance.branch_advance_id for advance in evidence.branch_advances
        }
        branch_advance_dependency_ids = {
            dependency_id
            for advance in evidence.branch_advances
            for dependency_id in advance.exact_dependency_ids
        }
        sourced = {
            source_id
            for derivative in ordered
            for source_id in derivative.direct_source_ids
            if source_id in derivative_ids
        }
        return cls.mint(
            launch_subject_ids=tuple(sorted(set(launch_subject_ids))),
            evidence=evidence,
            derivatives=ordered,
            terminal_derivative_ids=tuple(sorted(derivative_ids - sourced)),
            exact_dependency_ids=tuple(
                sorted(
                    set(launch_subject_ids)
                    | derivative_ids
                    | branch_advance_ids
                    | branch_advance_dependency_ids
                    | {evidence.evidence_id}
                    | {
                        derivative.authorization_safety_state_id
                        for derivative in ordered
                    }
                )
            ),
        )

    @property
    def derivative_ids(self) -> tuple[str, ...]:
        return tuple(item.derivative_id for item in self.derivatives)


@dataclass(frozen=True)
class RunDerivativeTaint(StrictContract):
    """One transitive security taint path into a run derivative."""

    taint_id: str
    derivative_id: str
    root_revoked_subject_id: str
    revocation_id: str
    predecessor_taint_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-derivative-taint"
    IDENTITY_FIELD: ClassVar[str] = "taint_id"

    def _validate(self) -> None:
        for value, name in (
            (self.derivative_id, "run taint derivative"),
            (self.root_revoked_subject_id, "run taint root subject"),
            (self.revocation_id, "run taint revocation"),
        ):
            require_content_id(value, name)
        _require_sorted_content_ids(
            self.predecessor_taint_ids,
            "run taint predecessors",
        )
        if self.taint_id in self.predecessor_taint_ids:
            raise ResumeContractError("run derivative taint cannot source itself")


def propagate_derivative_taints(
    frontier: RunDerivativeFrontier,
    security_observation: SecurityDenylistObservation,
) -> tuple[RunDerivativeTaint, ...]:
    """Derive the exact fixed-point taint closure for one authenticated snapshot."""

    if (
        type(frontier) is not RunDerivativeFrontier
        or type(security_observation) is not SecurityDenylistObservation
    ):
        raise ResumeContractError(
            "run taint propagation requires typed frontier and security observation"
        )
    derivatives = {item.derivative_id: item for item in frontier.derivatives}
    all_direct_sources = {
        source_id
        for derivative in frontier.derivatives
        for source_id in derivative.direct_source_ids
    }
    taints: dict[tuple[str, str], RunDerivativeTaint] = {}
    revocations = {
        revocation.revocation_id: revocation
        for revocation in security_observation.matched_revocations
    }
    changed = True
    while changed:
        changed = False
        for derivative_id in sorted(derivatives):
            derivative = derivatives[derivative_id]
            for revocation_id in sorted(revocations):
                revocation = revocations[revocation_id]
                key = (derivative_id, revocation_id)
                if key in taints:
                    continue
                predecessor_taints = tuple(
                    sorted(
                        taints[(source_id, revocation_id)].taint_id
                        for source_id in derivative.direct_source_ids
                        if (source_id, revocation_id) in taints
                    )
                )
                launch_root_revoked = (
                    revocation.subject_id in frontier.launch_subject_ids
                )
                run_wide_authority_revoked = (
                    revocation.subject_id not in derivatives
                    and revocation.subject_id not in all_direct_sources
                )
                if (
                    not launch_root_revoked
                    and not run_wide_authority_revoked
                    and derivative_id != revocation.subject_id
                    and revocation.subject_id not in derivative.direct_source_ids
                    and not predecessor_taints
                ):
                    continue
                taints[key] = RunDerivativeTaint.mint(
                    derivative_id=derivative_id,
                    root_revoked_subject_id=revocation.subject_id,
                    revocation_id=revocation_id,
                    predecessor_taint_ids=predecessor_taints,
                )
                changed = True
    return tuple(sorted(taints.values(), key=lambda item: item.taint_id))


def resume_security_subject_ids(
    *,
    bootstrap_pin: BootstrapPin,
    release_use_observation: ExpertReleaseUsePolicyObservation,
    derivative_frontier: RunDerivativeFrontier,
    predecessor_safety_state_id: str | None,
    inherited_security_subject_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the complete subject closure for one resumed dangerous boundary."""

    if (
        type(bootstrap_pin) is not BootstrapPin
        or type(release_use_observation) is not ExpertReleaseUsePolicyObservation
        or type(derivative_frontier) is not RunDerivativeFrontier
    ):
        raise ResumeContractError(
            "resume subjects require typed pin, policy, and derivative frontier"
        )
    _require_sorted_content_ids(
        inherited_security_subject_ids,
        "inherited run security subjects",
    )
    subjects = {
        bootstrap_pin.bootstrap_pin_id,
        bootstrap_pin.installation_receipt.workspace_installation_receipt_id,
        bootstrap_pin.launch_manifest.launch_manifest_id,
        *bootstrap_pin.launch_manifest.exact_dependency_ids,
        derivative_frontier.frontier_id,
        *derivative_frontier.exact_dependency_ids,
        *_release_use_subject_ids(release_use_observation),
        *inherited_security_subject_ids,
    }
    if predecessor_safety_state_id is not None:
        require_content_id(
            predecessor_safety_state_id,
            "predecessor run safety state",
        )
        subjects.add(predecessor_safety_state_id)
    return tuple(sorted(subjects))


@dataclass(frozen=True)
class RunSafetyState(StrictContract):
    """Authenticated safety and release-use disposition for one exact frontier."""

    safety_state_id: str
    predecessor_safety_state_id: str | None
    predecessor_taint_ids: tuple[str, ...]
    inherited_security_subject_ids: tuple[str, ...]
    bootstrap_pin: BootstrapPin
    boundary: RunSafetyBoundary
    boundary_sequence: int
    derivative_frontier: RunDerivativeFrontier
    security_observation: SecurityDenylistObservation
    release_use_observation: ExpertReleaseUsePolicyObservation
    release_use_mode: RunReleaseUseMode
    derivative_taints: tuple[RunDerivativeTaint, ...]
    disposition: RunEligibilityDisposition
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-safety-state"
    IDENTITY_FIELD: ClassVar[str] = "safety_state_id"

    def _validate(self) -> None:
        if type(self.bootstrap_pin) is not BootstrapPin:
            raise ResumeContractError(
                "run safety state requires one embedded bootstrap pin"
            )
        _require_sorted_content_ids(
            self.predecessor_taint_ids,
            "run safety predecessor taints",
        )
        _require_sorted_content_ids(
            self.inherited_security_subject_ids,
            "run safety inherited security subjects",
        )
        if self.predecessor_safety_state_id is None:
            installation = self.bootstrap_pin.installation_receipt
            expected_initial_branches = {
                installation.workspace_git_branch: (
                    installation.workspace_baseline_commit_sha
                )
            }
            if (
                self.predecessor_taint_ids
                or self.inherited_security_subject_ids
                or self.boundary_sequence != 0
                or self.derivative_frontier.derivatives
                or any(
                    revision != 0
                    for revision in self.derivative_frontier.evidence.state_authority_revisions.values()
                )
                or self.derivative_frontier.evidence.artifact_digests
                or self.derivative_frontier.evidence.branch_advances
                or dict(self.derivative_frontier.evidence.branch_origin_heads)
                != expected_initial_branches
                or dict(self.derivative_frontier.evidence.branch_heads)
                != expected_initial_branches
                or self.boundary is not RunSafetyBoundary.INITIALIZATION
            ):
                raise ResumeContractError(
                    "initial run safety state must have an empty predecessor frontier"
                )
        else:
            require_content_id(
                self.predecessor_safety_state_id,
                "run safety predecessor",
            )
            if self.boundary_sequence <= 0:
                raise ResumeContractError(
                    "successor run safety state requires a positive sequence"
                )
            if self.boundary is RunSafetyBoundary.INITIALIZATION:
                raise ResumeContractError(
                    "successor run safety state cannot initialize the run"
                )
        if type(self.boundary_sequence) is not int or self.boundary_sequence < 0:
            raise ResumeContractError(
                "run safety boundary sequence must be non-negative"
            )
        if (
            type(self.derivative_frontier) is not RunDerivativeFrontier
            or type(self.security_observation) is not SecurityDenylistObservation
            or type(self.release_use_observation)
            is not ExpertReleaseUsePolicyObservation
        ):
            raise ResumeContractError(
                "run safety state uses an unrecognized typed authority"
            )
        taint_ids = tuple(item.taint_id for item in self.derivative_taints)
        if taint_ids != tuple(sorted(set(taint_ids))):
            raise ResumeContractError("run derivative taints must be sorted and unique")
        expected_taints = propagate_derivative_taints(
            self.derivative_frontier,
            self.security_observation,
        )
        if self.derivative_taints != expected_taints:
            raise ResumeContractError(
                "run derivative taints are not the exact security fixed point"
            )
        expected_disposition = (
            RunEligibilityDisposition.SECURITY_BLOCKED
            if self.security_observation.matched_revocations
            else (
                RunEligibilityDisposition.REPRODUCIBILITY_ONLY
                if (
                    self.release_use_observation.matched_revocations
                    or self.release_use_mode is RunReleaseUseMode.PINNED_OFFLINE
                )
                else RunEligibilityDisposition.ELIGIBLE
            )
        )
        if self.disposition is not expected_disposition:
            raise ResumeContractError(
                "run safety disposition differs from its observations"
            )
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "run safety exact dependencies",
            required=True,
        )
        expected_dependencies = {
            self.bootstrap_pin.launch_manifest.launch_manifest_id,
            self.bootstrap_pin.bootstrap_pin_id,
            self.derivative_frontier.frontier_id,
            self.security_observation.observation_id,
            self.release_use_observation.observation_id,
            *self.predecessor_taint_ids,
            *self.inherited_security_subject_ids,
            *(item.taint_id for item in self.derivative_taints),
            *(item.revocation_id for item in self.derivative_taints),
        }
        if self.predecessor_safety_state_id is not None:
            expected_dependencies.add(self.predecessor_safety_state_id)
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ResumeContractError("run safety dependency closure is not exact")
        self._require_bootstrap_join()

    @classmethod
    def build(
        cls,
        *,
        predecessor: "RunSafetyState | None",
        bootstrap_pin: BootstrapPin,
        boundary: RunSafetyBoundary,
        derivative_frontier: RunDerivativeFrontier,
        security_observation: SecurityDenylistObservation,
        release_use_observation: ExpertReleaseUsePolicyObservation,
        release_use_mode: RunReleaseUseMode,
    ) -> "RunSafetyState":
        if predecessor is not None and type(predecessor) is not RunSafetyState:
            raise ResumeContractError(
                "run safety predecessor must be one exact safety state"
            )
        if (
            predecessor is not None
            and predecessor.disposition is RunEligibilityDisposition.SECURITY_BLOCKED
        ):
            raise ResumeContractError(
                "security-blocked run safety state cannot authorize a successor"
            )
        if predecessor is not None:
            predecessor.require_bootstrap_pin(bootstrap_pin)
        predecessor_id = None if predecessor is None else predecessor.safety_state_id
        predecessor_taint_ids = (
            ()
            if predecessor is None
            else tuple(taint.taint_id for taint in predecessor.derivative_taints)
        )
        inherited_security_subject_ids = (
            ()
            if predecessor is None
            else predecessor.security_observation.checked_subject_ids
        )
        taints = propagate_derivative_taints(
            derivative_frontier,
            security_observation,
        )
        disposition = (
            RunEligibilityDisposition.SECURITY_BLOCKED
            if security_observation.matched_revocations
            else (
                RunEligibilityDisposition.REPRODUCIBILITY_ONLY
                if (
                    release_use_observation.matched_revocations
                    or release_use_mode is RunReleaseUseMode.PINNED_OFFLINE
                )
                else RunEligibilityDisposition.ELIGIBLE
            )
        )
        dependencies = {
            bootstrap_pin.launch_manifest.launch_manifest_id,
            bootstrap_pin.bootstrap_pin_id,
            derivative_frontier.frontier_id,
            security_observation.observation_id,
            release_use_observation.observation_id,
            *predecessor_taint_ids,
            *inherited_security_subject_ids,
            *(item.taint_id for item in taints),
            *(item.revocation_id for item in taints),
        }
        if predecessor_id is not None:
            dependencies.add(predecessor_id)
        state = cls.mint(
            predecessor_safety_state_id=predecessor_id,
            predecessor_taint_ids=predecessor_taint_ids,
            inherited_security_subject_ids=inherited_security_subject_ids,
            bootstrap_pin=bootstrap_pin,
            boundary=boundary,
            boundary_sequence=(
                0 if predecessor is None else predecessor.boundary_sequence + 1
            ),
            derivative_frontier=derivative_frontier,
            security_observation=security_observation,
            release_use_observation=release_use_observation,
            release_use_mode=release_use_mode,
            derivative_taints=taints,
            disposition=disposition,
            exact_dependency_ids=tuple(sorted(dependencies)),
        )
        state.require_predecessor(predecessor)
        state.require_bootstrap_pin(bootstrap_pin)
        return state

    def require_predecessor(self, predecessor: "RunSafetyState | None") -> None:
        if predecessor is not None and type(predecessor) is not RunSafetyState:
            raise ResumeContractError(
                "run safety predecessor must be one exact safety state"
            )
        expected_id = None if predecessor is None else predecessor.safety_state_id
        expected_taints = (
            ()
            if predecessor is None
            else tuple(taint.taint_id for taint in predecessor.derivative_taints)
        )
        expected_sequence = (
            0 if predecessor is None else predecessor.boundary_sequence + 1
        )
        expected_inherited_subjects = (
            ()
            if predecessor is None
            else predecessor.security_observation.checked_subject_ids
        )
        if (
            self.predecessor_safety_state_id != expected_id
            or self.predecessor_taint_ids != expected_taints
            or self.inherited_security_subject_ids != expected_inherited_subjects
            or self.boundary_sequence != expected_sequence
        ):
            raise ResumeContractError(
                "run safety state does not preserve its exact predecessor"
            )
        if predecessor is None:
            return
        current_derivatives = {
            derivative.derivative_id: derivative
            for derivative in self.derivative_frontier.derivatives
        }
        predecessor_derivative_ids = set(predecessor.derivative_frontier.derivative_ids)
        predecessor_evidence = predecessor.derivative_frontier.evidence
        current_evidence = self.derivative_frontier.evidence
        predecessor_branch_advances = {
            item.branch_advance_id: item
            for item in predecessor_evidence.branch_advances
        }
        current_branch_advances = {
            item.branch_advance_id: item for item in current_evidence.branch_advances
        }
        prior_release_revocations = {
            item.revocation_id
            for item in predecessor.release_use_observation.matched_revocations
        }
        current_release_revocations = {
            item.revocation_id
            for item in self.release_use_observation.matched_revocations
        }
        prior_security_revocations = {
            item.revocation_id
            for item in predecessor.security_observation.matched_revocations
        }
        current_security_revocations = {
            item.revocation_id for item in self.security_observation.matched_revocations
        }
        if (
            self.bootstrap_pin != predecessor.bootstrap_pin
            or self.derivative_frontier.launch_subject_ids
            != predecessor.derivative_frontier.launch_subject_ids
            or any(
                current_derivatives.get(derivative.derivative_id) != derivative
                for derivative in predecessor.derivative_frontier.derivatives
            )
            or any(
                derivative.authorization_safety_state_id != predecessor.safety_state_id
                for derivative in self.derivative_frontier.derivatives
                if derivative.derivative_id not in predecessor_derivative_ids
            )
            or set(current_evidence.state_authority_digests)
            != set(predecessor_evidence.state_authority_digests)
            or any(
                current_evidence.state_authority_revisions[authority]
                < predecessor_evidence.state_authority_revisions[authority]
                or (
                    current_evidence.state_authority_revisions[authority]
                    == predecessor_evidence.state_authority_revisions[authority]
                    and current_evidence.state_authority_digests[authority]
                    != predecessor_evidence.state_authority_digests[authority]
                )
                for authority in predecessor_evidence.state_authority_digests
            )
            or any(
                current_evidence.artifact_digests.get(locator) != digest
                for locator, digest in predecessor_evidence.artifact_digests.items()
            )
            or current_evidence.branch_origin_heads
            != predecessor_evidence.branch_origin_heads
            or set(current_evidence.branch_heads)
            != set(predecessor_evidence.branch_heads)
            or any(
                current_branch_advances.get(branch_advance_id) != branch_advance
                for branch_advance_id, branch_advance in predecessor_branch_advances.items()
            )
            or any(
                advance.authorization_safety_state_id != predecessor.safety_state_id
                for branch_advance_id, advance in current_branch_advances.items()
                if branch_advance_id not in predecessor_branch_advances
            )
            or self.release_use_observation.catalog_generation
            < predecessor.release_use_observation.catalog_generation
            or (
                self.release_use_observation.catalog_generation
                == predecessor.release_use_observation.catalog_generation
                and _release_use_authority_coordinates(self.release_use_observation)
                != _release_use_authority_coordinates(
                    predecessor.release_use_observation
                )
            )
            or not prior_release_revocations.issubset(current_release_revocations)
            or self.security_observation.generation
            < predecessor.security_observation.generation
            or (
                self.security_observation.generation
                == predecessor.security_observation.generation
                and _security_authority_coordinates(self.security_observation)
                != _security_authority_coordinates(predecessor.security_observation)
            )
            or not prior_security_revocations.issubset(current_security_revocations)
        ):
            raise ResumeContractError(
                "run safety successor changed or rolled back durable history"
            )

    def require_bootstrap_pin(self, bootstrap_pin: BootstrapPin) -> None:
        if (
            type(bootstrap_pin) is not BootstrapPin
            or self.bootstrap_pin != bootstrap_pin
        ):
            raise ResumeContractError(
                "run safety state requires one exact bootstrap pin"
            )
        self._require_bootstrap_join()

    def _require_bootstrap_join(self) -> None:
        bootstrap_pin = self.bootstrap_pin
        manifest = bootstrap_pin.launch_manifest
        release_use = self.release_use_observation
        security = self.security_observation
        pinned_release_revocation_ids = {
            item.revocation_id
            for item in manifest.release_use_observation.matched_revocations
        }
        current_release_revocation_ids = {
            item.revocation_id for item in release_use.matched_revocations
        }
        expected_launch_subjects = tuple(
            sorted(
                {
                    bootstrap_pin.bootstrap_pin_id,
                    bootstrap_pin.installation_receipt.workspace_installation_receipt_id,
                    manifest.launch_manifest_id,
                    *manifest.exact_dependency_ids,
                }
            )
        )
        expected_security_subjects = resume_security_subject_ids(
            bootstrap_pin=bootstrap_pin,
            release_use_observation=release_use,
            derivative_frontier=self.derivative_frontier,
            predecessor_safety_state_id=self.predecessor_safety_state_id,
            inherited_security_subject_ids=(self.inherited_security_subject_ids),
        )
        if (
            self.derivative_frontier.launch_subject_ids != expected_launch_subjects
            or release_use.scope_id != manifest.scope_contract.scope_id
            or release_use.scope_contract_id
            != manifest.scope_contract.scope_contract_id
            or release_use.scope_repository_binding_hash
            != manifest.scope_repository_binding_hash
            or release_use.repository_full_name
            != manifest.scope_repositories.knowledge_repository
            or release_use.repository_node_id
            != manifest.knowledge_component.publication.repository_node_id
            or release_use.checked_release_ids != (manifest.expert_manifest.release_id,)
            or release_use.catalog_generation
            < manifest.release_use_observation.catalog_generation
            or (
                release_use.catalog_generation
                == manifest.release_use_observation.catalog_generation
                and _release_use_authority_coordinates(release_use)
                != _release_use_authority_coordinates(manifest.release_use_observation)
            )
            or not pinned_release_revocation_ids.issubset(
                current_release_revocation_ids
            )
            or any(
                revocation.release_publication_id
                != manifest.expert_component.publication.publication_id
                or revocation.release_activation_witness_id
                != manifest.expert_component.activation_witness.witness_id
                for revocation in release_use.matched_revocations
            )
            or (
                self.release_use_mode is RunReleaseUseMode.PINNED_OFFLINE
                and release_use != manifest.release_use_observation
            )
            or security.scope_id != manifest.scope_contract.scope_id
            or security.scope_contract_id != manifest.scope_contract.scope_contract_id
            or security.scope_repository_binding_hash
            != manifest.scope_repository_binding_hash
            or security.repository_full_name
            != manifest.scope_repositories.security_repository
            or security.repository_node_id
            != manifest.security_observation.repository_node_id
            or security.generation < manifest.security_observation.generation
            or (
                security.generation == manifest.security_observation.generation
                and _security_authority_coordinates(security)
                != _security_authority_coordinates(manifest.security_observation)
            )
            or security.checked_subject_ids != expected_security_subjects
        ):
            raise ResumeContractError(
                "run safety state does not join its bootstrap authority"
            )


__all__ = [
    "propagate_derivative_taints",
    "resume_security_subject_ids",
    "ResumeContractError",
    "RunBranchAdvance",
    "RunDerivativeEvidence",
    "RunDerivativeFrontier",
    "RunDerivativeKind",
    "RunDerivativeRecord",
    "RunDerivativeTaint",
    "RunEligibilityDisposition",
    "RunReleaseUseMode",
    "RunSafetyBoundary",
    "RunSafetyState",
]
