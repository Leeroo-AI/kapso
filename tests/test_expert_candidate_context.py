from dataclasses import fields, replace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    CrossRunTaskBindingSettings,
    ExpertRepositoryMap,
)
from kapso.cross_run.expert.candidate_context import (
    ExpertCandidateContextError,
    ExpertCandidateReplayEvidence,
    ExpertCandidateValidationContext,
    project_agent_candidate_validation_context,
)
from kapso.cross_run.expert.triggers import (
    ExpertSourceBaseTreeReceipt,
    ExpertTriggerEvaluator,
)
from test_cross_run_retrieval import (
    analogical_context,
    outcome_episodes,
    source_fixture,
)
from test_expert_triggers import (
    clone_episode,
    supported_claim,
    trigger_packet,
    trigger_settings,
)


def _digest(label: str) -> str:
    return tree_or_blob_digest(label.encode("utf-8"))


def _remint(record, **changes):
    payload = {
        field.name: getattr(record, field.name)
        for field in fields(record)
        if field.name != record.IDENTITY_FIELD
    }
    payload.update(changes)
    return type(record).mint(**payload)


def _projected_context():
    settings = trigger_settings()
    _, context, episode, _, claim, _, _ = source_fixture()
    positive = outcome_episodes(episode)[0]
    first = clone_episode(
        positive,
        name="context-first",
        campaign_id="campaign-context-first",
        context=context,
    )
    second = clone_episode(
        positive,
        name="context-second",
        campaign_id="campaign-context-second",
        context=analogical_context(context),
    )
    supported = supported_claim(claim, (first, second))
    packet = trigger_packet(
        settings=settings,
        episodes=(first, second),
        claims=(supported,),
    )
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)
    context = project_agent_candidate_validation_context(
        packet=packet,
        decision=decision,
    )
    return packet, decision, context


def _replay_dependencies(
    *,
    manifests,
    scopes,
    episodes,
    evidence_authority_ids,
    proof_reference_ids,
):
    return tuple(
        sorted(
            {
                *(manifest.snapshot_id for manifest in manifests),
                *(scope.scope_contract_id for scope in scopes),
                *(episode.episode_id for episode in episodes),
                *(episode.source_bundle_id for episode in episodes),
                *evidence_authority_ids,
                *proof_reference_ids,
            }
        )
    )


def _remint_replay(replay, **changes):
    values = {
        "knowledge_snapshot_manifests": replay.knowledge_snapshot_manifests,
        "scope_contracts": replay.scope_contracts,
        "episodes": replay.episodes,
        "causal_episode_ids": replay.causal_episode_ids,
        "causal_episode_reason_codes": replay.causal_episode_reason_codes,
        "evidence_authority_ids": replay.evidence_authority_ids,
        "proof_reference_ids": replay.proof_reference_ids,
    }
    values.update(changes)
    values["stable_dependency_ids"] = changes.get(
        "stable_dependency_ids",
        _replay_dependencies(
            manifests=values["knowledge_snapshot_manifests"],
            scopes=values["scope_contracts"],
            episodes=values["episodes"],
            evidence_authority_ids=values["evidence_authority_ids"],
            proof_reference_ids=values["proof_reference_ids"],
        ),
    )
    return ExpertCandidateReplayEvidence.mint(**values)


def _context_dependencies(context, *, scope_contract=None, replay_evidence=None):
    scope = scope_contract or context.scope_contract
    replay = replay_evidence or context.replay_evidence
    dependencies = {
        scope.scope_contract_id,
        replay.replay_evidence_id,
        *replay.stable_dependency_ids,
    }
    if context.source_base_release is not None:
        dependencies.update(
            {
                context.source_base_scope_contract.scope_contract_id,
                context.source_base_release.release_id,
                context.source_base_tree_receipt.source_base_tree_receipt_id,
                context.source_base_tree_receipt.source_extraction_receipt.extraction_receipt_id,
                context.source_base_repository_map.repository_map_id,
                *(
                    module.module_contract_id
                    for module in context.source_base_module_contracts
                ),
            }
        )
    return tuple(sorted(dependencies))


def _receipt_with_archive(context, *, archive_ref, archive_digest):
    original = context.source_base_tree_receipt
    cache = replace(
        original.cache_verification_receipt,
        asset_digests={archive_ref: archive_digest},
    )
    extraction = _remint(
        original.source_extraction_receipt,
        source_archive_ref=archive_ref,
        source_archive_digest=archive_digest,
    )
    return ExpertSourceBaseTreeReceipt.mint(
        release_id=original.release_id,
        cache_verification_receipt=cache,
        source_extraction_receipt=extraction,
        source_base_tree_hash=original.source_base_tree_hash,
        repository_map_id=original.repository_map_id,
        module_contract_ids=original.module_contract_ids,
        materializer_version=original.materializer_version,
    )


def test_agent_context_projection_is_deterministic_and_exact():
    packet, decision, context = _projected_context()
    repeated = project_agent_candidate_validation_context(
        packet=packet,
        decision=decision,
    )

    assert repeated == context
    assert repeated.validation_context_id == context.validation_context_id
    assert context.scope_contract == packet.scope_contract
    assert context.source_base_release == packet.source_base_release
    assert context.source_base_tree_receipt == packet.source_base_tree_receipt
    assert context.source_base_repository_map == packet.source_base_repository_map
    assert context.source_base_module_contracts == packet.source_base_module_contracts
    assert context.replay_evidence.evidence_authority_ids == tuple(
        sorted(
            (
                packet.knowledge_snapshot_manifest.snapshot_id,
                packet.evidence_packet_id,
                decision.trigger_decision_id,
            )
        )
    )


def test_bootstrap_context_requires_the_canonical_empty_parent():
    settings = trigger_settings()
    packet = trigger_packet(settings=settings, bootstrap=True)
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)
    context = project_agent_candidate_validation_context(
        packet=packet,
        decision=decision,
    )

    assert context.source_base_release is None
    with pytest.raises(ExpertCandidateContextError, match="explicit empty source base"):
        _remint(context, source_base_tree_hash=_digest("not-empty"))


def test_context_rejects_replay_from_another_domain_scope():
    settings = trigger_settings()
    packet = trigger_packet(settings=settings, bootstrap=True)
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)
    context = project_agent_candidate_validation_context(
        packet=packet,
        decision=decision,
    )
    foreign_scope = _remint(context.scope_contract, scope_id="foreign_domain")
    original_binding = context.active_task_bindings[0]
    foreign_binding = CrossRunTaskBindingSettings(
        scope_id=foreign_scope.scope_id,
        task_family_id=original_binding.task_family_id,
        task_adapter_id=original_binding.task_adapter_id,
    )

    with pytest.raises(ExpertCandidateContextError, match="replay scope leaves"):
        _remint(
            context,
            scope_contract=foreign_scope,
            active_task_bindings=(foreign_binding,),
            stable_dependency_ids=_context_dependencies(
                context,
                scope_contract=foreign_scope,
            ),
        )


def test_released_context_joins_exact_map_and_modules():
    _, _, context = _projected_context()
    repository_map = context.source_base_repository_map
    mismatched_map = ExpertRepositoryMap.mint(
        scope_contract_id=repository_map.scope_contract_id,
        capability_nodes=repository_map.capability_nodes,
        dependency_edges=repository_map.dependency_edges,
        task_adapter_boundary=repository_map.task_adapter_boundary,
        validation_entrypoints=repository_map.validation_entrypoints,
        architecture_invariants=tuple(
            sorted(
                (
                    *repository_map.architecture_invariants,
                    "A substituted map invariant.",
                )
            )
        ),
    )

    with pytest.raises(ExpertCandidateContextError, match="source-base authority"):
        _remint(context, source_base_repository_map=mismatched_map)
    with pytest.raises(ExpertCandidateContextError, match="source-base authority"):
        _remint(context, source_base_module_contracts=())


@pytest.mark.parametrize(
    ("archive_ref", "archive_digest"),
    (
        (None, _digest("substituted archive")),
        ("other-source.tar.zst", _digest("other archive")),
    ),
    ids=("archive-digest", "extraction-reference"),
)
def test_released_context_joins_exact_archive_and_extraction(
    archive_ref,
    archive_digest,
):
    _, _, context = _projected_context()
    receipt = _receipt_with_archive(
        context,
        archive_ref=archive_ref or context.source_base_release.source_archive_ref,
        archive_digest=archive_digest,
    )

    with pytest.raises(ExpertCandidateContextError, match="release closure"):
        _remint(context, source_base_tree_receipt=receipt)


def test_released_context_accepts_only_equal_or_directly_superseding_scope():
    settings = trigger_settings()
    parent_packet = trigger_packet(settings=settings)
    source_base_scope = parent_packet.scope_contract
    successor = _remint(
        source_base_scope,
        supersedes_scope_contract_id=source_base_scope.scope_contract_id,
        purpose=source_base_scope.purpose + " with one attested revision",
    )
    packet = trigger_packet(
        settings=settings,
        current_scope_contract=successor,
        released_scope_contract=source_base_scope,
    )
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)
    context = project_agent_candidate_validation_context(
        packet=packet,
        decision=decision,
    )

    assert context.scope_contract == successor
    assert context.source_base_scope_contract == source_base_scope

    indirect_successor = _remint(
        successor,
        supersedes_scope_contract_id=successor.scope_contract_id,
        purpose=successor.purpose + " followed by another revision",
    )
    with pytest.raises(ExpertCandidateContextError, match="directly supersede"):
        _remint(
            context,
            scope_contract=indirect_successor,
            stable_dependency_ids=_context_dependencies(
                context,
                scope_contract=indirect_successor,
            ),
        )


def test_replay_evidence_preserves_exact_admitted_episode_lineage():
    _, _, context = _projected_context()
    replay = context.replay_evidence
    parent = replay.episodes[0]
    child = clone_episode(
        parent,
        name="lineage-child",
        campaign_id=parent.source["campaign_id"],
        context=parent.task_context_binding,
        parent_episode_ref=parent.episode_id,
    )
    original_manifest = replay.knowledge_snapshot_manifests[0]
    manifest = _remint(
        original_manifest,
        admitted_episode_ids=(child.episode_id,),
        active_claim_revision_ids=(),
        proof_dependency_closure_ids=tuple(
            sorted(
                {
                    *original_manifest.proof_dependency_closure_ids,
                    child.episode_id,
                }
            )
        ),
    )
    authorities = tuple(
        sorted(
            {
                manifest.snapshot_id,
                *(
                    authority_id
                    for authority_id in replay.evidence_authority_ids
                    if authority_id != original_manifest.snapshot_id
                ),
            }
        )
    )
    lineage = _remint_replay(
        replay,
        knowledge_snapshot_manifests=(manifest,),
        episodes=tuple(sorted((parent, child), key=lambda item: item.episode_id)),
        causal_episode_ids=(child.episode_id,),
        causal_episode_reason_codes={child.episode_id: ("causal_trigger_evidence",)},
        evidence_authority_ids=authorities,
    )

    assert {episode.episode_id for episode in lineage.episodes} == {
        parent.episode_id,
        child.episode_id,
    }
    with pytest.raises(ExpertCandidateContextError, match="omits an admitted"):
        _remint_replay(
            lineage,
            episodes=(child,),
        )


def test_replay_snapshots_cannot_cross_supply_bundle_authority():
    _, _, context = _projected_context()
    replay = context.replay_evidence
    original_manifest = replay.knowledge_snapshot_manifests[0]
    admitted_without_bundle = _remint(
        original_manifest,
        included_bundle_ids=(),
    )
    bundle_without_admission = _remint(
        original_manifest,
        parent_snapshot_ids=(admitted_without_bundle.snapshot_id,),
        admitted_episode_ids=(),
        active_claim_revision_ids=(),
    )
    manifests = tuple(
        sorted(
            (admitted_without_bundle, bundle_without_admission),
            key=lambda manifest: manifest.snapshot_id,
        )
    )
    evidence_authority_ids = tuple(
        sorted(
            {
                *(manifest.snapshot_id for manifest in manifests),
                *(
                    authority_id
                    for authority_id in replay.evidence_authority_ids
                    if authority_id != original_manifest.snapshot_id
                ),
            }
        )
    )

    with pytest.raises(ExpertCandidateContextError, match="independently authorize"):
        _remint_replay(
            replay,
            knowledge_snapshot_manifests=manifests,
            evidence_authority_ids=evidence_authority_ids,
        )


def test_replay_rejects_episode_scope_and_proof_substitution():
    _, _, context = _projected_context()
    replay = context.replay_evidence
    manifest = replay.knowledge_snapshot_manifests[0]
    foreign_scope = _remint(
        replay.scope_contracts[0],
        purpose=replay.scope_contracts[0].purpose + " from another revision",
    )
    foreign_manifest = _remint(
        manifest,
        scope_contract_id=foreign_scope.scope_contract_id,
    )
    authorities = tuple(
        sorted(
            {
                foreign_manifest.snapshot_id,
                *(
                    authority_id
                    for authority_id in replay.evidence_authority_ids
                    if authority_id != manifest.snapshot_id
                ),
            }
        )
    )
    with pytest.raises(ExpertCandidateContextError, match="independently authorize"):
        _remint_replay(
            replay,
            knowledge_snapshot_manifests=(foreign_manifest,),
            scope_contracts=(foreign_scope,),
            evidence_authority_ids=authorities,
        )

    missing_proof = replay.episodes[0].sanitation_report_id
    assert missing_proof in replay.proof_reference_ids
    with pytest.raises(ExpertCandidateContextError, match="omits episode proof"):
        _remint_replay(
            replay,
            proof_reference_ids=tuple(
                proof_id
                for proof_id in replay.proof_reference_ids
                if proof_id != missing_proof
            ),
        )


def test_replay_authorities_and_context_dependencies_are_exact():
    _, _, context = _projected_context()
    replay = context.replay_evidence
    snapshot_id = replay.knowledge_snapshot_manifests[0].snapshot_id
    without_snapshot = tuple(
        authority_id
        for authority_id in replay.evidence_authority_ids
        if authority_id != snapshot_id
    )

    with pytest.raises(ExpertCandidateContextError, match="omit a knowledge snapshot"):
        _remint_replay(
            replay,
            evidence_authority_ids=without_snapshot,
        )

    extra_dependency = content_id("fixture", {"dependency": "unearned"})
    with pytest.raises(ExpertCandidateContextError, match="dependency closure"):
        _remint(
            context,
            stable_dependency_ids=tuple(
                sorted((*context.stable_dependency_ids, extra_dependency))
            ),
        )

    alternate_binding = CrossRunTaskBindingSettings(
        scope_id=context.scope_id,
        task_family_id="relational_tabular_prediction",
        task_adapter_id="relbench",
    )
    changed = _remint(
        context,
        active_task_bindings=(alternate_binding,),
    )
    assert changed.validation_context_id != context.validation_context_id
