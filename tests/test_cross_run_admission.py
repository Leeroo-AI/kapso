import json
from dataclasses import replace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.catalog.admission import AdmissionReducer
from kapso.cross_run.record_contracts import (
    CatalogAgentOperationRecord,
    CatalogRevocation,
    CatalogTaint,
    ClaimEvidenceClosure,
    SANITATION_REPORT_SCHEMA,
    SANITATION_SCANNER_VERSION,
    SanitationReport,
    catalog_agent_operation_id,
)
from kapso.cross_run.contracts import (
    AdmissionState,
    ArtifactEnvironment,
    BundleArtifactRef,
    CodingAgentOperationReceipt,
    ComparisonStatus,
    EffectUncertaintyMethod,
    EpisodeEvaluationStatus,
    EvaluationFingerprint,
    ExpertScopeContract,
    ExecutionStatus,
    InterventionStructure,
    IdentityConflictError,
    KnowledgeClaim,
    MissingReferenceError,
    ObjectiveDirection,
    PriorIdea,
    PriorIdeaStatus,
    RelativeEffect,
    ReviewAssertion,
    TaskContextBinding,
    TransferAttempt,
    TransferEpisode,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.agent_artifacts import (
    CODING_AGENT_ARTIFACT_FILENAMES as ARTIFACT_FILENAMES,
)
from test_cross_run_contracts import build_records

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def fixture_id(name):
    return content_id("admission-test-fixture", {"name": name})


def digest(name):
    return tree_or_blob_digest(name.encode("utf-8"))


def catalog_settings():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).catalog


def scope_contract():
    return next(
        record for record in build_records() if isinstance(record, ExpertScopeContract)
    )


def context(name="shared"):
    return TaskContextBinding.mint(
        scope_contract_id=scope_contract().scope_contract_id,
        scope_id="ml_ai",
        task_family_id="language_model_post_training",
        task_adapter_id="posttrain",
        capability_tags=("training",),
        input_contract_fingerprint=digest(f"{name}-input"),
        target_contract_fingerprint=digest(f"{name}-target"),
        starting_artifact_refs=(),
        method_fingerprint=digest(f"{name}-method"),
        toolchain_fingerprint=digest("toolchain"),
        dependency_runtime_fingerprint=digest("runtime"),
        budget_hardware_envelope={"accelerator": "test"},
        transfer_dimensions={
            "dataset_family": name,
            "runtime_family": "test_runtime",
        },
    )


def environment():
    return ArtifactEnvironment.mint(
        kapso_commit="0" * 40,
        expert_base_release_id=fixture_id("expert-release"),
        task_adapter_hash=digest("task-adapter"),
        dependency_lock_hash=digest("dependency-lock"),
    )


def sanitation_report(source_bundle_id, task_context):
    return SanitationReport.mint(
        schema=SANITATION_REPORT_SCHEMA,
        capture_manifest_id=source_bundle_id,
        scope_id=task_context.scope_id,
        task_family_id=task_context.task_family_id,
        policy_version="sanitation-policy-v1",
        policy_fingerprint=digest("sanitation-policy"),
        scanner_version=SANITATION_SCANNER_VERSION,
        status="admitted",
        findings=(),
        excluded_paths=(),
        taint_sources=(),
        admitted_refs={"capture.json": digest(source_bundle_id)},
    )


def evaluation_fingerprint(name):
    return EvaluationFingerprint.mint(
        benchmark_id="benchmark",
        dataset_version="v1",
        split_version="validation",
        evaluator_fingerprint=digest("evaluator"),
        metric_name="score",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        fidelity="full",
        fraction=1.0,
        seed_or_replicate_ids=("seed_0",),
        aggregation_protocol="mean",
        judge_version=None,
    )


def completed_attempt(name, candidate_value=0.8, *, isolated=True):
    fingerprint = evaluation_fingerprint(name)
    effect = RelativeEffect(
        evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
        metric_name="score",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        candidate_value=candidate_value,
        source_parent_value=0.7,
        raw_delta=candidate_value - 0.7,
        normalized_delta=candidate_value - 0.7,
        uncertainty=None,
        uncertainty_method=EffectUncertaintyMethod.UNAVAILABLE,
    )
    return TransferAttempt(
        execution_revision=0,
        captured_at="2026-07-21T12:00:00Z",
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.VALID,
        evaluation_fingerprints=(fingerprint,),
        score_of_record_fingerprint_id=fingerprint.evaluation_fingerprint_id,
        comparison_status=ComparisonStatus.COMPARABLE,
        measurements={"score": candidate_value},
        source_parent_effect=effect,
        intervention_ref=BundleArtifactRef(
            relative_path=f"branches/{name}.tar.zst",
            checksum=digest(f"{name}-branch"),
        ),
        intervention_structure=(
            InterventionStructure.ISOLATED_BY_ABLATION
            if isolated
            else InterventionStructure.COUPLED
        ),
        feedback=(),
        technical_difficulties=(),
        confounders=(),
    )


def technical_attempt():
    return TransferAttempt(
        execution_revision=0,
        captured_at="2026-07-21T12:00:00Z",
        execution_status=ExecutionStatus.FAILED_TECHNICAL,
        evaluation_status=EpisodeEvaluationStatus.NOT_RUN,
        evaluation_fingerprints=(),
        score_of_record_fingerprint_id=None,
        comparison_status=ComparisonStatus.NOT_COMPARABLE,
        measurements={},
        source_parent_effect=None,
        intervention_ref=None,
        intervention_structure=InterventionStructure.UNDETERMINED,
        feedback=(),
        technical_difficulties=("The worker exited before evaluation.",),
        confounders=(),
    )


def episode(
    name,
    run_id,
    *,
    attempt=None,
    task_context=None,
    supersedes=None,
    derivation_refs=None,
):
    bundle_id = fixture_id(f"{name}-bundle")
    event_id = fixture_id(f"{name}-event")
    resolved_context = task_context or context()
    sanitation_id = sanitation_report(bundle_id, resolved_context).report_id
    return TransferEpisode.mint(
        source={
            "scope_id": "ml_ai",
            "run_id": run_id,
            "campaign_id": "campaign",
            "node_id": f"node_{name}",
            "idea_id": f"idea_{name}",
            "batch_id": "batch",
        },
        source_bundle_id=bundle_id,
        supersedes_projection_id=supersedes,
        task_context_binding=resolved_context,
        artifact_environment=environment(),
        proposal=f"Evaluate intervention {name}.",
        parent_episode_ref=None,
        attempts=(attempt or completed_attempt(name),),
        terminal_attempt_revision=0,
        safe_observation_refs=(),
        sanitation_report_id=sanitation_id,
        derivation_refs=tuple(sorted(derivation_refs or (event_id,))),
    )


def successor_episode(predecessor, *, name="successor"):
    bundle_id = fixture_id(f"{name}-bundle")
    return TransferEpisode.mint(
        source=dict(predecessor.source),
        source_bundle_id=bundle_id,
        supersedes_projection_id=predecessor.episode_id,
        task_context_binding=predecessor.task_context_binding,
        artifact_environment=predecessor.artifact_environment,
        proposal=f"Re-evaluate intervention {name}.",
        parent_episode_ref=None,
        attempts=(completed_attempt(name),),
        terminal_attempt_revision=0,
        safe_observation_refs=(),
        sanitation_report_id=sanitation_report(
            bundle_id,
            predecessor.task_context_binding,
        ).report_id,
        derivation_refs=(fixture_id(f"{name}-event"),),
    )


def prior_idea(name):
    bundle_id = fixture_id(f"{name}-bundle")
    task_context = context()
    return PriorIdea.mint(
        source_bundle_id=bundle_id,
        supersedes_projection_id=None,
        source={
            "scope_id": "ml_ai",
            "run_id": "run_prior",
            "campaign_id": "campaign",
            "batch_id": "batch",
            "idea_id": f"idea_{name}",
        },
        proposal=f"Consider {name} later.",
        descriptor={
            "approach_family": "family",
            "expected_effect": "unknown",
            "intervention_target": "pipeline",
            "mechanism": "hypothesis",
        },
        assumptions=(),
        source_status=PriorIdeaStatus.REJECTED,
        source_rationale="The local selector rejected it.",
        source_evidence_refs=(),
        task_context_binding=task_context,
        sanitation_report_id=sanitation_report(bundle_id, task_context).report_id,
    )


def claim(name, support, contradictions=(), *, supersedes=(), evaluated=None):
    settings = catalog_settings()
    classified = (*support, *contradictions)
    evaluated_records = evaluated if evaluated is not None else classified
    support_ids = {item.episode_id for item in support}
    contradiction_ids = {item.episode_id for item in contradictions}
    assessments = tuple(
        {
            "episode_id": item.episode_id,
            "relationship": (
                "support"
                if item.episode_id in support_ids
                else (
                    "contradiction"
                    if item.episode_id in contradiction_ids
                    else "not_applicable"
                )
            ),
            "rationale": "The fixture classifies this exact episode.",
        }
        for item in sorted(evaluated_records, key=lambda record: record.episode_id)
    )
    raw_claim = {
        "statement": f"Claim {name} is applicable under its exact predicates.",
        "mechanism": "The isolated intervention changes the measured mechanism.",
        "applicability_predicates": {"dataset_family": "shared"},
        "explicit_exclusions": [
            "Contexts outside the registered predicate are excluded."
        ],
        "evidence_assessments": list(assessments),
        "supersedes_revision_ids": list(sorted(supersedes)),
    }
    final_output = json.dumps({"claims": [raw_claim]}, sort_keys=True)
    preimage = {
        "packet": {
            "scope_contract": scope_contract().to_dict(),
            "episodes": tuple(
                {"episode_id": item.episode_id}
                for item in sorted(
                    evaluated_records,
                    key=lambda record: record.episode_id,
                )
            ),
        },
        "template": f"claim-fixture-{name}",
        "schema": {"type": "object"},
        "catalog_configuration": settings.to_dict(),
    }
    operation = CodingAgentOperationReceipt.mint(
        operation_id=catalog_agent_operation_id(preimage),
        principal_id=settings.claim_proposer_id,
        role=settings.claim_proposer_role,
        cli=settings.claim_proposer.cli,
        model=settings.claim_proposer.model,
        effort=settings.claim_proposer.effort,
        artifact_checksums={
            filename: (
                tree_or_blob_digest(final_output.encode("utf-8"))
                if filename == "final.json"
                else digest(f"claim-{name}-{filename}")
            )
            for filename in ARTIFACT_FILENAMES
        },
    )
    packet_digest = tree_or_blob_digest(canonical_json_bytes(preimage["packet"]))
    claim_id = (
        "claim_"
        + content_id(
            "claim-lineage",
            {
                "operation_receipt_id": operation.operation_receipt_id,
                "proposal_ordinal": 0,
            },
        ).rsplit(":", 1)[1][:32]
    )
    knowledge_claim = KnowledgeClaim.mint(
        claim_id=claim_id,
        scope_contract_id=scope_contract().scope_contract_id,
        statement=raw_claim["statement"],
        mechanism=raw_claim["mechanism"],
        applicability_predicates=raw_claim["applicability_predicates"],
        explicit_exclusions=tuple(sorted(raw_claim["explicit_exclusions"])),
        supporting_episode_ids=tuple(sorted(support_ids)),
        contradicting_episode_ids=tuple(sorted(contradiction_ids)),
        proposal_provenance={
            "operation_receipt_id": operation.operation_receipt_id,
            "packet_digest": packet_digest,
            "proposal_ordinal": 0,
        },
        supersedes_revision_ids=tuple(sorted(supersedes)),
    )
    evidence_closure = ClaimEvidenceClosure.mint(
        claim_revision_id=knowledge_claim.revision_id,
        evaluated_episode_ids=tuple(
            sorted(item.episode_id for item in evaluated_records)
        ),
        supporting_episode_ids=knowledge_claim.supporting_episode_ids,
        contradicting_episode_ids=knowledge_claim.contradicting_episode_ids,
        evidence_assessments=assessments,
        proposer_operation_receipt_id=operation.operation_receipt_id,
        packet_digest=packet_digest,
    )
    operation_record = CatalogAgentOperationRecord.mint(
        operation_kind="claim_proposal",
        operation_receipt_id=operation.operation_receipt_id,
        operation_preimage=preimage,
        final_output=final_output,
        produced_object_ids=tuple(
            sorted(
                (
                    knowledge_claim.revision_id,
                    evidence_closure.claim_evidence_closure_id,
                )
            )
        ),
    )
    return knowledge_claim, evidence_closure, operation, operation_record


def review_facts(subject, evidence, *, settings_override=None):
    settings = settings_override or catalog_settings()
    receipts = []
    assertions = []
    operations = []
    evidence_ids = tuple(sorted(item.episode_id for item in evidence))
    for position, reviewer in enumerate(settings.reviewers):
        output = {
            "judgment": settings.admission.approval_judgment,
            "rationale": "The complete evidence meets the current rubric.",
            "exact_evidence_refs": list(evidence_ids),
            "supersedes_assertion_id": None,
        }
        final_output = json.dumps(output, sort_keys=True)
        preimage = {
            "packet": {
                "subject": {"record_id": subject},
                "evidence_records": tuple(
                    {"record_id": evidence_id} for evidence_id in evidence_ids
                ),
            },
            "template": f"review-fixture-{position}",
            "schema": {"type": "object"},
            "reviewer": reviewer.to_dict(),
        }
        operation = CodingAgentOperationReceipt.mint(
            operation_id=catalog_agent_operation_id(preimage),
            principal_id=reviewer.reviewer_id,
            role=reviewer.reviewer_role,
            cli=reviewer.agent.cli,
            model=reviewer.agent.model,
            effort=reviewer.agent.effort,
            artifact_checksums={
                filename: (
                    tree_or_blob_digest(final_output.encode("utf-8"))
                    if filename == "final.json"
                    else digest(f"{subject}-{position}-{filename}")
                )
                for filename in ARTIFACT_FILENAMES
            },
        )
        receipts.append(operation)
        assertion = ReviewAssertion.mint(
            subject_id=subject,
            reviewer_id=reviewer.reviewer_id,
            reviewer_role=reviewer.reviewer_role,
            rubric_version=reviewer.rubric_version,
            judgment=output["judgment"],
            rationale=output["rationale"],
            exact_evidence_refs=evidence_ids,
            supersedes_assertion_id=None,
            review_operation_ref=operation.operation_receipt_id,
        )
        assertions.append(assertion)
        operations.append(
            CatalogAgentOperationRecord.mint(
                operation_kind="catalog_review",
                operation_receipt_id=operation.operation_receipt_id,
                operation_preimage=preimage,
                final_output=final_output,
                produced_object_ids=(assertion.assertion_id,),
            )
        )
    return tuple(assertions), tuple(receipts), tuple(operations)


def proof_ids(*projections):
    values = set()
    for projection in projections:
        values.add(projection.source_bundle_id)
        values.add(projection.sanitation_report_id)
        if isinstance(projection, TransferEpisode):
            values.update(projection.derivation_refs)
    return tuple(sorted(values))


def reports_for(*projections):
    reports = {}
    for projection in projections:
        report = sanitation_report(
            projection.source_bundle_id,
            projection.task_context_binding,
        )
        if report.report_id != projection.sanitation_report_id:
            raise AssertionError("projection sanitation fixture is inconsistent")
        reports[projection.sanitation_report_id] = report
    return tuple(reports[report_id] for report_id in sorted(reports))


def state_by_subject(reduction):
    return {state.subject_payload_id: state for state in reduction.states}


def test_admission_uses_provenance_and_quorum_without_outcome_sign_or_recency():
    first = episode(
        "positive", "run_positive", attempt=completed_attempt("positive", 0.8)
    )
    second = episode(
        "negative", "run_negative", attempt=completed_attempt("negative", 0.6)
    )
    technical = episode("technical", "run_technical", attempt=technical_attempt())
    deferred = prior_idea("deferred")
    knowledge_claim, evidence_closure, proposer_operation, proposal_record = claim(
        "direction_neutral",
        (first, second),
        evaluated=(first, second, technical),
    )
    assertions, receipts, review_operations = review_facts(
        knowledge_claim.revision_id,
        (first, second, technical),
    )
    projections = (first, second, technical, deferred)
    sanitation_reports = reports_for(*projections)
    reducer = AdmissionReducer(catalog_settings(), scope_contract())

    forward = reducer.reduce(
        catalog_generation=1,
        episodes=(first, second, technical),
        prior_ideas=(deferred,),
        claims=(knowledge_claim,),
        assertions=assertions,
        receipts=(proposer_operation, *receipts),
        operation_records=(proposal_record, *review_operations),
        claim_evidence_closures=(evidence_closure,),
        sanitation_reports=sanitation_reports,
        proof_object_ids=proof_ids(*projections),
    )
    shuffled = reducer.reduce(
        catalog_generation=1,
        episodes=(technical, second, first),
        prior_ideas=(deferred,),
        claims=(knowledge_claim,),
        assertions=tuple(reversed(assertions)),
        receipts=tuple(reversed((proposer_operation, *receipts))),
        operation_records=tuple(reversed((proposal_record, *review_operations))),
        claim_evidence_closures=(evidence_closure,),
        sanitation_reports=tuple(reversed(sanitation_reports)),
        proof_object_ids=tuple(reversed(proof_ids(*projections))),
    )

    assert tuple(state.to_json_bytes() for state in forward.states) == tuple(
        state.to_json_bytes() for state in shuffled.states
    )
    states = state_by_subject(forward)
    assert all(
        state.admission_state is AdmissionState.ADMITTED for state in states.values()
    )
    assert evidence_closure.not_applicable_episode_ids == (technical.episode_id,)


def test_claim_review_cannot_hide_a_not_applicable_episode_from_admission():
    first = episode("closure_support_a", "run_closure_support_a")
    second = episode("closure_support_b", "run_closure_support_b")
    technical = episode(
        "closure_technical", "run_closure_technical", attempt=technical_attempt()
    )
    knowledge_claim, closure, proposer_receipt, proposal_operation = claim(
        "complete_review_closure",
        (first, second),
        evaluated=(first, second, technical),
    )
    assertions, reviewer_receipts, reviewer_operations = review_facts(
        knowledge_claim.revision_id,
        (first, second),
    )

    with pytest.raises(IdentityConflictError, match="omits part"):
        AdmissionReducer(catalog_settings(), scope_contract()).reduce(
            catalog_generation=1,
            episodes=(first, second, technical),
            prior_ideas=(),
            claims=(knowledge_claim,),
            assertions=assertions,
            receipts=(proposer_receipt, *reviewer_receipts),
            operation_records=(proposal_operation, *reviewer_operations),
            claim_evidence_closures=(closure,),
            sanitation_reports=reports_for(first, second, technical),
            proof_object_ids=proof_ids(first, second, technical),
        )


def test_rejected_or_missing_sanitation_fact_cannot_admit_projection():
    original = episode("rejected_sanitation", "run_rejected")
    rejected_report = SanitationReport.mint(
        schema=SANITATION_REPORT_SCHEMA,
        capture_manifest_id=original.source_bundle_id,
        scope_id=original.task_context_binding.scope_id,
        task_family_id=original.task_context_binding.task_family_id,
        policy_version="sanitation-policy-v1",
        policy_fingerprint=digest("sanitation-policy"),
        scanner_version=SANITATION_SCANNER_VERSION,
        status="rejected",
        findings=(
            {
                "code": "openai_key",
                "evidence_digest": digest("secret-finding"),
                "path": "source.py",
                "severity": "reject",
            },
        ),
        excluded_paths=(),
        taint_sources=(),
        admitted_refs={},
    )
    payload = original.to_dict()
    payload.pop("episode_id")
    payload["sanitation_report_id"] = rejected_report.report_id
    rejected = TransferEpisode.mint(**payload)
    reducer = AdmissionReducer(catalog_settings(), scope_contract())

    with pytest.raises(MissingReferenceError):
        reducer.reduce(
            catalog_generation=1,
            episodes=(rejected,),
            prior_ideas=(),
            claims=(),
            assertions=(),
            receipts=(),
            operation_records=(),
            claim_evidence_closures=(),
            sanitation_reports=(),
            proof_object_ids=proof_ids(rejected),
        )

    reduction = reducer.reduce(
        catalog_generation=1,
        episodes=(rejected,),
        prior_ideas=(),
        claims=(),
        assertions=(),
        receipts=(),
        operation_records=(),
        claim_evidence_closures=(),
        sanitation_reports=(rejected_report,),
        proof_object_ids=proof_ids(rejected),
    )
    assert reduction.states[0].admission_state is AdmissionState.QUARANTINED


def test_claim_requires_independent_runs_isolation_and_complete_evidence_closure():
    first = episode("same_run_a", "run_shared")
    second = episode("same_run_b", "run_shared")
    knowledge_claim, evidence_closure, proposer_operation, proposal_record = claim(
        "insufficient_diversity", (first, second)
    )
    assertions, receipts, review_operations = review_facts(
        knowledge_claim.revision_id, (first, second)
    )
    reducer = AdmissionReducer(catalog_settings(), scope_contract())
    reduction = reducer.reduce(
        catalog_generation=1,
        episodes=(first, second),
        prior_ideas=(),
        claims=(knowledge_claim,),
        assertions=assertions,
        receipts=(proposer_operation, *receipts),
        operation_records=(proposal_record, *review_operations),
        claim_evidence_closures=(evidence_closure,),
        sanitation_reports=reports_for(first, second),
        proof_object_ids=proof_ids(first, second),
    )
    assert (
        state_by_subject(reduction)[knowledge_claim.revision_id].admission_state
        is AdmissionState.QUARANTINED
    )

    with pytest.raises(MissingReferenceError):
        reducer.reduce(
            catalog_generation=1,
            episodes=(first, second),
            prior_ideas=(),
            claims=(knowledge_claim,),
            assertions=assertions,
            receipts=(proposer_operation, *receipts),
            operation_records=(proposal_record, *review_operations),
            claim_evidence_closures=(),
            sanitation_reports=reports_for(first, second),
            proof_object_ids=proof_ids(first, second),
        )

    coupled = episode(
        "coupled",
        "run_coupled",
        attempt=completed_attempt("coupled", isolated=False),
    )
    isolated_claim, isolated_closure, isolated_operation, isolated_record = claim(
        "requires_isolation", (first, coupled)
    )
    coupled_assertions, coupled_receipts, coupled_review_operations = review_facts(
        isolated_claim.revision_id,
        (first, coupled),
    )
    coupled_reduction = reducer.reduce(
        catalog_generation=1,
        episodes=(first, coupled),
        prior_ideas=(),
        claims=(isolated_claim,),
        assertions=coupled_assertions,
        receipts=(isolated_operation, *coupled_receipts),
        operation_records=(isolated_record, *coupled_review_operations),
        claim_evidence_closures=(isolated_closure,),
        sanitation_reports=reports_for(first, coupled),
        proof_object_ids=proof_ids(first, coupled),
    )
    assert (
        state_by_subject(coupled_reduction)[isolated_claim.revision_id].admission_state
        is AdmissionState.QUARANTINED
    )


def test_claim_support_must_satisfy_its_registered_applicability_predicates():
    first = episode(
        "predicate_a",
        "run_predicate_a",
        task_context=context("outside_predicate"),
    )
    second = episode(
        "predicate_b",
        "run_predicate_b",
        task_context=context("outside_predicate"),
    )
    knowledge_claim, closure, proposer_receipt, proposal_operation = claim(
        "predicate_mismatch",
        (first, second),
    )
    assertions, reviewer_receipts, reviewer_operations = review_facts(
        knowledge_claim.revision_id,
        (first, second),
    )

    reduction = AdmissionReducer(catalog_settings(), scope_contract()).reduce(
        catalog_generation=1,
        episodes=(first, second),
        prior_ideas=(),
        claims=(knowledge_claim,),
        assertions=assertions,
        receipts=(proposer_receipt, *reviewer_receipts),
        operation_records=(proposal_operation, *reviewer_operations),
        claim_evidence_closures=(closure,),
        sanitation_reports=reports_for(first, second),
        proof_object_ids=proof_ids(first, second),
    )

    assert (
        state_by_subject(reduction)[knowledge_claim.revision_id].admission_state
        is AdmissionState.QUARANTINED
    )


def test_authenticated_claim_receipt_cannot_be_replayed_for_an_altered_claim():
    first = episode("replayed_claim_a", "run_replayed_claim_a")
    second = episode("replayed_claim_b", "run_replayed_claim_b")
    original, closure, receipt, operation = claim(
        "replayed_claim",
        (first, second),
    )
    claim_payload = original.to_dict()
    claim_payload.pop("revision_id")
    claim_payload["statement"] = "This altered statement was not emitted by the agent."
    forged = KnowledgeClaim.mint(**claim_payload)
    closure_payload = closure.to_dict()
    closure_payload.pop("claim_evidence_closure_id")
    closure_payload["claim_revision_id"] = forged.revision_id
    forged_closure = ClaimEvidenceClosure.mint(**closure_payload)
    replayed_operation = CatalogAgentOperationRecord.mint(
        operation_kind=operation.operation_kind,
        operation_receipt_id=operation.operation_receipt_id,
        operation_preimage=operation.operation_preimage,
        final_output=operation.final_output,
        produced_object_ids=tuple(
            sorted((forged.revision_id, forged_closure.claim_evidence_closure_id))
        ),
    )

    with pytest.raises(IdentityConflictError, match="authenticated model output"):
        AdmissionReducer(catalog_settings(), scope_contract()).reduce(
            catalog_generation=1,
            episodes=(first, second),
            prior_ideas=(),
            claims=(forged,),
            assertions=(),
            receipts=(receipt,),
            operation_records=(replayed_operation,),
            claim_evidence_closures=(forged_closure,),
            sanitation_reports=reports_for(first, second),
            proof_object_ids=proof_ids(first, second),
        )


def test_rotated_reviewer_cannot_review_its_own_historical_claim_proposal():
    first = episode("historical_self_review_a", "run_historical_self_review_a")
    second = episode("historical_self_review_b", "run_historical_self_review_b")
    knowledge_claim, closure, proposer_receipt, proposal_operation = claim(
        "historical_self_review",
        (first, second),
    )
    original_settings = catalog_settings()
    historical_proposer_as_reviewer = replace(
        original_settings.reviewers[0],
        reviewer_id=original_settings.claim_proposer_id,
    )
    rotated_settings = replace(
        original_settings,
        claim_proposer_id="rotated_catalog_claim_proposer",
        reviewers=tuple(
            sorted(
                (
                    historical_proposer_as_reviewer,
                    original_settings.reviewers[1],
                ),
                key=lambda reviewer: reviewer.reviewer_id,
            )
        ),
    )
    assertions, reviewer_receipts, reviewer_operations = review_facts(
        knowledge_claim.revision_id,
        (first, second),
        settings_override=rotated_settings,
    )

    with pytest.raises(IdentityConflictError, match="historical output"):
        AdmissionReducer(rotated_settings, scope_contract()).reduce(
            catalog_generation=1,
            episodes=(first, second),
            prior_ideas=(),
            claims=(knowledge_claim,),
            assertions=assertions,
            receipts=(proposer_receipt, *reviewer_receipts),
            operation_records=(proposal_operation, *reviewer_operations),
            claim_evidence_closures=(closure,),
            sanitation_reports=reports_for(first, second),
            proof_object_ids=proof_ids(first, second),
        )


def test_revoked_contradiction_taints_claim_instead_of_strengthening_it():
    first = episode("support_a", "run_support_a")
    second = episode("support_b", "run_support_b")
    contradiction = episode("contradiction", "run_contradiction")
    knowledge_claim, evidence_closure, proposer_operation, proposal_record = claim(
        "with_contradiction",
        (first, second),
        (contradiction,),
    )
    assertions, receipts, review_operations = review_facts(
        knowledge_claim.revision_id,
        (first, second, contradiction),
    )
    revocation_evidence = fixture_id("contamination-finding")
    revocation = CatalogRevocation.mint(
        subject_id=contradiction.episode_id,
        reason_code="evaluation_contamination",
        rationale="The episode included evaluation-only information.",
        exact_evidence_refs=(revocation_evidence,),
    )
    projections = (first, second, contradiction)
    reduction = AdmissionReducer(catalog_settings(), scope_contract()).reduce(
        catalog_generation=1,
        episodes=projections,
        prior_ideas=(),
        claims=(knowledge_claim,),
        assertions=assertions,
        receipts=(proposer_operation, *receipts),
        operation_records=(proposal_record, *review_operations),
        claim_evidence_closures=(evidence_closure,),
        sanitation_reports=reports_for(*projections),
        proof_object_ids=tuple(sorted((*proof_ids(*projections), revocation_evidence))),
        revocations=(revocation,),
    )
    states = state_by_subject(reduction)

    assert states[contradiction.episode_id].admission_state is AdmissionState.REVOKED
    assert states[knowledge_claim.revision_id].admission_state is AdmissionState.REVOKED
    assert (
        revocation.revocation_id in states[knowledge_claim.revision_id].taint_source_ids
    )


def test_taint_propagates_only_over_proof_edges_not_supersession_lineage():
    predecessor = episode("lineage", "run_lineage")
    successor = successor_episode(predecessor)
    first_support = episode("other_support", "run_other")
    knowledge_claim, evidence_closure, proposer_operation, proposal_record = claim(
        "clean_successor", (successor, first_support)
    )
    assertions, receipts, review_operations = review_facts(
        knowledge_claim.revision_id,
        (successor, first_support),
    )
    finding = fixture_id("late-finding")
    revocation = CatalogRevocation.mint(
        subject_id=predecessor.episode_id,
        reason_code="late_contamination",
        rationale="A late scanner found contamination in the old projection.",
        exact_evidence_refs=(finding,),
    )
    projections = (predecessor, successor, first_support)
    reducer = AdmissionReducer(catalog_settings(), scope_contract())
    reduction = reducer.reduce(
        catalog_generation=1,
        episodes=projections,
        prior_ideas=(),
        claims=(knowledge_claim,),
        assertions=assertions,
        receipts=(proposer_operation, *receipts),
        operation_records=(proposal_record, *review_operations),
        claim_evidence_closures=(evidence_closure,),
        sanitation_reports=reports_for(*projections),
        proof_object_ids=tuple(sorted((*proof_ids(*projections), finding))),
        revocations=(revocation,),
    )
    states = state_by_subject(reduction)

    assert states[predecessor.episode_id].admission_state is AdmissionState.REVOKED
    assert states[successor.episode_id].admission_state is AdmissionState.ADMITTED
    assert (
        states[knowledge_claim.revision_id].admission_state is AdmissionState.ADMITTED
    )

    taint_evidence = fixture_id("taint-evidence")
    taint = CatalogTaint.mint(
        subject_id=successor.episode_id,
        source_subject_id=predecessor.episode_id,
        reason_code="derived_copy",
        rationale="The successor copied the contaminated proof material.",
        exact_evidence_refs=(taint_evidence,),
    )
    tainted = reducer.reduce(
        catalog_generation=1,
        episodes=projections,
        prior_ideas=(),
        claims=(knowledge_claim,),
        assertions=assertions,
        receipts=(proposer_operation, *receipts),
        operation_records=(proposal_record, *review_operations),
        claim_evidence_closures=(evidence_closure,),
        sanitation_reports=reports_for(*projections),
        proof_object_ids=tuple(
            sorted((*proof_ids(*projections), finding, taint_evidence))
        ),
        revocations=(revocation,),
        taints=(taint,),
    )
    tainted_states = state_by_subject(tainted)
    assert (
        tainted_states[successor.episode_id].admission_state is AdmissionState.REVOKED
    )
    assert (
        tainted_states[knowledge_claim.revision_id].admission_state
        is AdmissionState.REVOKED
    )
    assert (
        taint.taint_id in tainted_states[knowledge_claim.revision_id].taint_source_ids
    )


def test_predecessor_state_is_linked_while_current_facts_are_fully_reduced():
    first = episode("predecessor_state", "run_predecessor")
    reducer = AdmissionReducer(catalog_settings(), scope_contract())
    initial = reducer.reduce(
        catalog_generation=1,
        episodes=(first,),
        prior_ideas=(),
        claims=(),
        assertions=(),
        receipts=(),
        operation_records=(),
        claim_evidence_closures=(),
        sanitation_reports=reports_for(first),
        proof_object_ids=proof_ids(first),
    )
    finding = fixture_id("generation-two-finding")
    revocation = CatalogRevocation.mint(
        subject_id=first.episode_id,
        reason_code="generation_two_revocation",
        rationale="The next generation received a new contamination finding.",
        exact_evidence_refs=(finding,),
    )
    current = reducer.reduce(
        catalog_generation=2,
        episodes=(first,),
        prior_ideas=(),
        claims=(),
        assertions=(),
        receipts=(),
        operation_records=(),
        claim_evidence_closures=(),
        sanitation_reports=reports_for(first),
        proof_object_ids=tuple(sorted((*proof_ids(first), finding))),
        revocations=(revocation,),
        predecessor_states=(initial.states[0],),
    )

    assert (
        current.states[0].predecessor_state_id
        == initial.states[0].catalog_entry_state_id
    )
    assert current.states[0].admission_state is AdmissionState.REVOKED
    assert current.states[0].revocation_ids == (revocation.revocation_id,)
