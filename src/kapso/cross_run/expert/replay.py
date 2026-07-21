"""Policy-derived source replay selection for expert validation."""

from __future__ import annotations

from dataclasses import dataclass

from kapso.cross_run.canonical import require_identifier
from kapso.cross_run.contracts import (
    ExpertSourceReplayAdapterPackagePin,
    ExpertSourceReplayCase,
    ExpertSourceReplaySelection,
    MissingReferenceError,
    TransferEpisode,
)
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.expert.triggers import (
    ExpertEvolutionTriggerDecision,
    episode_lineage_id,
)
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
    packet = closure.trigger_packet
    decision = closure.trigger_decision
    candidate_module_contracts = closure.module_contracts
    policy = settings.policy.validation_policy()
    if (
        stored_candidate.commit_record.candidate_id != manifest.candidate_id
        or decision.evidence_packet_id != packet.evidence_packet_id
        or decision.knowledge_snapshot_id != packet.knowledge_snapshot_id
        or decision.configuration_fingerprint != packet.configuration_fingerprint
        or manifest.configuration_fingerprint != packet.configuration_fingerprint
        or manifest.trigger_evidence_packet_id != packet.evidence_packet_id
        or manifest.trigger_decision_id != decision.trigger_decision_id
        or not decision.candidate_required
    ):
        raise ExpertSourceReplayError(
            "source replay decision does not authorize the supplied packet"
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

    episodes_by_id = {episode.episode_id: episode for episode in packet.episodes}
    claims_by_id = {claim.revision_id: claim for claim in packet.claims}
    observations_by_id = {
        observation.observation_id: observation
        for observation in packet.trigger_observations
    }
    episodes_by_bundle: dict[str, list[TransferEpisode]] = {}
    for episode in packet.episodes:
        episodes_by_bundle.setdefault(episode.source_bundle_id, []).append(episode)

    pending_evidence = [
        (evidence_id, "decision") for evidence_id in decision.trigger_evidence_ids
    ]
    processed_evidence: set[tuple[str, str]] = set()
    causal_episode_ids: set[str] = set()
    episode_reasons: dict[str, set[str]] = {}
    selection_evidence_ids: set[str] = set()
    contradiction_trigger = decision.reason_code == "released_capability_contradiction"
    position = 0
    while position < len(pending_evidence):
        evidence_id, role = pending_evidence[position]
        position += 1
        evidence_edge = (evidence_id, role)
        if evidence_edge in processed_evidence:
            continue
        processed_evidence.add(evidence_edge)
        namespace = evidence_id.split(":sha256:", 1)[0]
        if namespace == "transfer-episode":
            if evidence_id not in episodes_by_id:
                raise MissingReferenceError(
                    "source replay trigger episode is absent from the packet"
                )
            if not contradiction_trigger or role == "contradicting_claim":
                causal_episode_ids.add(evidence_id)
                selection_evidence_ids.add(evidence_id)
                episode_reasons.setdefault(evidence_id, set()).add(
                    "causal_contradiction"
                    if role == "contradicting_claim"
                    else "causal_trigger_evidence"
                )
        elif namespace == "expert-trigger-observation":
            observation = observations_by_id.get(evidence_id)
            if observation is None:
                raise MissingReferenceError(
                    "source replay trigger observation is absent from the packet"
                )
            selection_evidence_ids.add(evidence_id)
            pending_evidence.extend(
                (child_id, "observation") for child_id in observation.exact_evidence_ids
            )
        elif namespace == "knowledge-claim-revision":
            claim = claims_by_id.get(evidence_id)
            if claim is None:
                raise MissingReferenceError(
                    "source replay trigger claim is absent from the packet"
                )
            selection_evidence_ids.add(evidence_id)
            if contradiction_trigger:
                pending_evidence.extend(
                    (episode_id, "contradicting_claim")
                    for episode_id in claim.contradicting_episode_ids
                )
        elif namespace == "run-bundle":
            bundle_episodes = episodes_by_bundle.get(evidence_id, ())
            if not bundle_episodes:
                raise MissingReferenceError(
                    "source replay trigger bundle has no packet episode"
                )
            if not contradiction_trigger or role == "contradicting_claim":
                selection_evidence_ids.add(evidence_id)
                for episode in bundle_episodes:
                    causal_episode_ids.add(episode.episode_id)
                    selection_evidence_ids.add(episode.episode_id)
                    episode_reasons.setdefault(episode.episode_id, set()).add(
                        "causal_trigger_bundle"
                    )

    _validate_causal_projections(
        causal_episode_ids,
        episodes_by_id,
        decision,
    )
    parent_contracts = {
        contract.module_id: contract for contract in packet.module_contracts
    }
    coverage_episode_ids: set[str] = set()
    for contract in candidate_module_contracts:
        parent = parent_contracts.get(contract.module_id)
        if (
            parent is not None
            and parent.module_contract_id == contract.module_contract_id
        ):
            continue
        support_ids = set(contract.supporting_episode_ids)
        failure_ids = set(contract.known_failure_episode_ids)
        if (support_ids | failure_ids) - set(episodes_by_id):
            raise MissingReferenceError(
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
    snapshot_bundle_ids = set(packet.knowledge_snapshot_manifest.included_bundle_ids)
    packet_proof_ids = set(packet.proof_reference_ids)
    if not selected_bundle_ids.issubset(snapshot_bundle_ids) or not (
        selected_bundle_ids.issubset(packet_proof_ids)
    ):
        raise MissingReferenceError(
            "selected source replay bundles leave the snapshot proof closure"
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
        context = episode.task_context_binding
        environment = episode.artifact_environment
        key = (
            context.scope_contract_id,
            context.task_family_id,
            context.task_adapter_id,
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
        packet.evidence_packet_id,
        decision.trigger_decision_id,
        packet.knowledge_snapshot_id,
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
        trigger_evidence_packet_id=packet.evidence_packet_id,
        trigger_decision_id=decision.trigger_decision_id,
        knowledge_snapshot_id=packet.knowledge_snapshot_id,
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


def _validate_causal_projections(
    causal_episode_ids: set[str],
    episodes_by_id: dict[str, TransferEpisode],
    decision: ExpertEvolutionTriggerDecision,
) -> None:
    causal_episodes = tuple(
        episodes_by_id[episode_id] for episode_id in causal_episode_ids
    )
    causal_context_ids = {
        episode.task_context_binding.task_context_binding_id
        for episode in causal_episodes
    }
    if decision.task_context_binding_ids and causal_context_ids != set(
        decision.task_context_binding_ids
    ):
        raise ExpertSourceReplayError(
            "source replay causal contexts differ from the trigger decision"
        )
    causal_lineage_ids = {
        episode_lineage_id(episode, episodes_by_id) for episode in causal_episodes
    }
    if decision.independent_lineage_ids and causal_lineage_ids != set(
        decision.independent_lineage_ids
    ):
        raise ExpertSourceReplayError(
            "source replay causal lineages differ from the trigger decision"
        )
