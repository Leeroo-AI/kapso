"""Policy-derived source replay selection for expert validation."""

from __future__ import annotations

from dataclasses import dataclass

from kapso.cross_run.canonical import require_identifier
from kapso.cross_run.contracts import (
    ExpertSourceReplayAdapterPackagePin,
    ExpertSourceReplayCase,
    ExpertSourceReplaySelection,
)
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.settings import ExpertValidationSettings


class ExpertSourceReplayError(ValueError):
    """Replay selection evidence is missing, ambiguous, or inconsistent."""


@dataclass(frozen=True)
class ExpertSourceReplaySelectionResult:
    selection: ExpertSourceReplaySelection | None
    reason_code: str

    def __post_init__(self) -> None:
        require_identifier(self.reason_code, "source replay selection reason_code")
        if (self.selection is None) != (self.reason_code != "selected"):
            raise ExpertSourceReplayError(
                "source replay selection result and reason must agree"
            )


def _derive_expert_source_replay_selection(
    *,
    stored_candidate: StoredExpertCandidate,
    settings: ExpertValidationSettings,
) -> ExpertSourceReplaySelectionResult:
    """Derive replay cases from one store-validated candidate and active policy."""

    closure = stored_candidate.closure
    manifest = closure.manifest
    context = closure.validation_context
    replay_evidence = context.replay_evidence
    candidate_module_contracts = closure.module_contracts
    policy = settings.policy.validation_policy()
    if (
        stored_candidate.commit_record.candidate_id != manifest.candidate_id
        or manifest.validation_context_ref != context.validation_context_id
        or manifest.scope_contract_id != context.scope_contract.scope_contract_id
        or manifest.source_base_release_id
        != (
            None
            if context.source_base_release is None
            else context.source_base_release.release_id
        )
    ):
        raise ExpertSourceReplayError(
            "source replay context does not authorize the supplied candidate"
        )
    candidate_module_ids = tuple(
        contract.module_id for contract in candidate_module_contracts
    )
    if candidate_module_ids != tuple(sorted(set(candidate_module_ids))):
        raise ExpertSourceReplayError(
            "candidate replay modules must be sorted and uniquely identified"
        )
    if manifest.module_contract_refs != tuple(
        sorted(contract.module_contract_id for contract in candidate_module_contracts)
    ):
        raise ExpertSourceReplayError(
            "candidate replay modules differ from the candidate manifest"
        )

    episodes_by_id = {
        episode.episode_id: episode for episode in replay_evidence.episodes
    }
    causal_episode_ids = set(replay_evidence.causal_episode_ids)
    episode_reasons = {
        episode_id: set(reasons)
        for episode_id, reasons in replay_evidence.causal_episode_reason_codes.items()
    }
    selection_evidence_ids = set(causal_episode_ids)
    source_base_contracts = {
        contract.module_id: contract for contract in context.source_base_module_contracts
    }
    coverage_episode_ids: set[str] = set()
    for contract in candidate_module_contracts:
        source_base_contract = source_base_contracts.get(contract.module_id)
        if (
            source_base_contract is not None
            and source_base_contract.module_contract_id == contract.module_contract_id
        ):
            continue
        support_ids = set(contract.supporting_episode_ids)
        failure_ids = set(contract.known_failure_episode_ids)
        if (support_ids | failure_ids) - set(episodes_by_id):
            raise ExpertSourceReplayError(
                "changed module replay evidence leaves the trigger packet"
            )
        if support_ids or failure_ids:
            selection_evidence_ids.add(contract.module_contract_id)
        for episode_id in support_ids:
            coverage_episode_ids.add(episode_id)
            episode_reasons.setdefault(episode_id, set()).add("changed_module_support")
        for episode_id in failure_ids:
            coverage_episode_ids.add(episode_id)
            episode_reasons.setdefault(episode_id, set()).add("changed_module_failure")

    selected_episode_ids = causal_episode_ids | coverage_episode_ids
    if not selected_episode_ids:
        return ExpertSourceReplaySelectionResult(
            selection=None,
            reason_code="source_replay_evidence_unavailable",
        )
    if any(
        episodes_by_id[episode_id].task_context_binding.scope_contract_id
        != manifest.scope_contract_id
        for episode_id in selected_episode_ids
    ):
        return ExpertSourceReplaySelectionResult(
            selection=None,
            reason_code="source_replay_scope_mapping_unavailable",
        )
    selected_bundle_ids = {
        episodes_by_id[episode_id].source_bundle_id
        for episode_id in selected_episode_ids
    }
    if (
        len(selected_episode_ids) > settings.policy.source_replay_episode_limit
        or len(selected_bundle_ids) > settings.policy.source_replay_bundle_limit
    ):
        return ExpertSourceReplaySelectionResult(
            selection=None,
            reason_code="source_replay_selection_limit_exceeded",
        )
    cases = tuple(
        ExpertSourceReplayCase(
            source_bundle_id=bundle_id,
            episode_ids=tuple(
                sorted(
                    episode_id
                    for episode_id in selected_episode_ids
                    if episodes_by_id[episode_id].source_bundle_id == bundle_id
                )
            ),
            episode_reason_codes={
                episode_id: tuple(sorted(episode_reasons[episode_id]))
                for episode_id in sorted(selected_episode_ids)
                if episodes_by_id[episode_id].source_bundle_id == bundle_id
            },
        )
        for bundle_id in sorted(selected_bundle_ids)
    )
    causal_ids = tuple(sorted(causal_episode_ids))
    coverage_ids = tuple(sorted(coverage_episode_ids - causal_episode_ids))
    adapter_episode_ids: dict[tuple[str, str, str, str, str], list[str]] = {}
    for episode_id in sorted(selected_episode_ids):
        episode = episodes_by_id[episode_id]
        task_context = episode.task_context_binding
        environment = episode.artifact_environment
        key = (
            task_context.scope_contract_id,
            task_context.task_family_id,
            task_context.task_adapter_id,
            environment.task_adapter_manifest_id,
            environment.task_adapter_verification_receipt_id,
        )
        adapter_episode_ids.setdefault(key, []).append(episode_id)
    source_adapter_pins = tuple(
        sorted(
            (
                ExpertSourceReplayAdapterPackagePin.mint(
                    scope_contract_id=scope_contract_id,
                    task_family_id=task_family_id,
                    task_adapter_id=task_adapter_id,
                    task_adapter_manifest_id=task_adapter_manifest_id,
                    verification_receipt_id=verification_receipt_id,
                    episode_ids=tuple(episode_ids),
                )
                for (
                    scope_contract_id,
                    task_family_id,
                    task_adapter_id,
                    task_adapter_manifest_id,
                    verification_receipt_id,
                ), episode_ids in adapter_episode_ids.items()
            ),
            key=lambda pin: pin.source_adapter_pin_id,
        )
    )
    dependencies = {
        manifest.candidate_id,
        stored_candidate.commit_record.commit_record_id,
        context.validation_context_id,
        *replay_evidence.evidence_authority_ids,
        *(
            snapshot.snapshot_id
            for snapshot in replay_evidence.knowledge_snapshot_manifests
        ),
        policy.validation_policy_id,
        *selection_evidence_ids,
        *selected_episode_ids,
        *selected_bundle_ids,
        *(pin.source_adapter_pin_id for pin in source_adapter_pins),
        *(pin.task_adapter_manifest_id for pin in source_adapter_pins),
        *(pin.verification_receipt_id for pin in source_adapter_pins),
    }
    selection = ExpertSourceReplaySelection.mint(
        candidate_id=manifest.candidate_id,
        candidate_tree_hash=manifest.candidate_tree_hash,
        candidate_commit_record_id=(stored_candidate.commit_record.commit_record_id),
        validation_context_id=context.validation_context_id,
        evidence_authority_ids=tuple(replay_evidence.evidence_authority_ids),
        validation_policy_id=policy.validation_policy_id,
        selection_policy_version=(
            settings.policy.source_replay_selection_policy_version
        ),
        configuration_fingerprint=settings.configuration_fingerprint,
        causal_episode_ids=causal_ids,
        coverage_episode_ids=coverage_ids,
        selection_evidence_ids=tuple(sorted(selection_evidence_ids)),
        cases=cases,
        source_adapter_pins=source_adapter_pins,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )
    return ExpertSourceReplaySelectionResult(
        selection=selection,
        reason_code="selected",
    )
