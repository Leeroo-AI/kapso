from dataclasses import replace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import ExpertValidationStage
from kapso.cross_run.expert.promotion_contracts import ExpertReleaseMatrixMode
from kapso.cross_run.expert.task_evaluation_contracts import (
    TASK_EVALUATION_REQUEST_CONTRACT_VERSION,
    TaskEvaluationCase,
    TaskEvaluationComputeBinding,
    TaskEvaluationContractError,
    TaskEvaluationExpertLeg,
    TaskEvaluationLegKind,
    TaskEvaluationRequest,
    TaskEvaluationReservation,
)
from kapso.cross_run.expert.task_evaluation_compute import (
    TaskEvaluationComputeError,
    derive_release_matrix_compute_bindings,
)
from kapso.cross_run.expert.task_evaluation_request import (
    PlanJoinedTaskEvaluationRequest,
    TaskEvaluationRequestPreparationError,
    prepare_task_evaluation_request,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    VerifiedTaskEvaluationCandidate,
    VerifiedTaskEvaluationParent,
)
from kapso.cross_run.expert.triggers import (
    ExpertParentTreeReceipt,
    ExpertTriggerEvidencePacket,
)
from test_expert_candidate_workspace import released_workspace_fixture
from test_expert_release_matrix_reservation import (
    _bootstrap_release_matrix_fixture,
    _release_matrix_fixture,
)


def _id(namespace: str, label: str) -> str:
    return content_id(namespace, {"label": label})


def _digest(label: str) -> str:
    return tree_or_blob_digest(label.encode("utf-8"))


def _remint(contract, **updates):
    values = contract.to_dict()
    values.pop(contract.IDENTITY_FIELD)
    values.update(updates)
    return type(contract).mint(**values)


def _leg(kind: TaskEvaluationLegKind) -> TaskEvaluationExpertLeg:
    if kind is TaskEvaluationLegKind.PARENT_CONTROL:
        artifact_id = _id("expert-base-release", "parent")
        receipt_id = _id("expert-parent-tree-receipt", "parent")
    else:
        artifact_id = _id("expert-candidate", "candidate")
        receipt_id = _id("expert-candidate-commit", "candidate")
    tree_label = (
        "parent-tree"
        if kind is TaskEvaluationLegKind.PARENT_CONTROL
        else "candidate-tree"
    )
    return TaskEvaluationExpertLeg.mint(
        kind=kind,
        expert_artifact_id=artifact_id,
        expert_source_receipt_id=receipt_id,
        expert_tree_hash=_digest(tree_label),
        exact_dependency_ids=tuple(sorted((artifact_id, receipt_id))),
    )


def _compute(
    leg_order: tuple[TaskEvaluationLegKind, ...],
) -> TaskEvaluationComputeBinding:
    return TaskEvaluationComputeBinding.mint(
        execution_protocol_version="kapso.task_evaluation.v1",
        execution_provider_id="kapso_task_evaluation_provider",
        execution_provider_version="kapso.task_evaluation_provider.v1",
        execution_provider_settings_digest=_digest("provider-settings"),
        sandbox_policy_version="kapso.task_evaluation_sandbox.v1",
        leg_wall_time_limit_seconds=300,
        termination_grace_seconds=10,
        cpu_millicore_limit=4000,
        memory_byte_limit=8_000_000_000,
        shared_memory_byte_limit=1_000_000_000,
        process_limit=512,
        open_file_limit=1024,
        writable_inode_limit=1000,
        writable_storage_byte_limit=1_000_000_000,
        output_entry_limit=100,
        output_byte_limit=100_000_000,
        stdout_byte_limit=10_000_000,
        stderr_byte_limit=10_000_000,
        accelerator_class_id=None,
        accelerator_count=0,
        leg_order=leg_order,
    )


def _case(
    mode: ExpertReleaseMatrixMode,
) -> TaskEvaluationCase:
    leg_kinds = (
        (TaskEvaluationLegKind.CANDIDATE,)
        if mode is ExpertReleaseMatrixMode.BOOTSTRAP
        else (
            TaskEvaluationLegKind.PARENT_CONTROL,
            TaskEvaluationLegKind.CANDIDATE,
        )
    )
    compute = _compute(leg_kinds)
    legs = tuple(sorted((_leg(kind) for kind in leg_kinds), key=lambda leg: leg.leg_id))
    adapter_authority_id = _id(
        "expert-release-matrix-adapter-authority",
        "adapter",
    )
    provenance_binding_id = _id(
        "expert-release-matrix-provenance-binding",
        "provenance",
    )
    release_matrix_case_id = _id("task-adapter-release-matrix-case", "case")
    context_id = _id("task-context-binding", "context")
    independence_id = _id(
        "task-adapter-release-matrix-independence-group",
        "independence",
    )
    cell_ids = tuple(
        sorted(
            (
                _id("expert-release-matrix-evaluation-cell", "quality"),
                _id("expert-release-matrix-evaluation-cell", "robustness"),
            )
        )
    )
    fingerprint_ids = tuple(
        sorted(
            (
                _id("evaluation-fingerprint", "quality"),
                _id("evaluation-fingerprint", "robustness"),
            )
        )
    )
    artifact_ids = (_id("task-adapter-release-matrix-starting-artifact", "fixture"),)
    dependencies = {
        adapter_authority_id,
        provenance_binding_id,
        release_matrix_case_id,
        context_id,
        independence_id,
        *cell_ids,
        *fingerprint_ids,
        *artifact_ids,
        compute.compute_binding_id,
        *(leg.leg_id for leg in legs),
        *(dependency_id for leg in legs for dependency_id in leg.exact_dependency_ids),
    }
    return TaskEvaluationCase.mint(
        adapter_authority_id=adapter_authority_id,
        provenance_binding_id=provenance_binding_id,
        release_matrix_case_id=release_matrix_case_id,
        task_context_binding_id=context_id,
        independence_group_id=independence_id,
        evaluation_cell_ids=cell_ids,
        evaluation_fingerprint_ids=fingerprint_ids,
        starting_artifact_ids=artifact_ids,
        compute_binding=compute,
        legs=legs,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )


def _request(mode: ExpertReleaseMatrixMode) -> TaskEvaluationRequest:
    case = _case(mode)
    plan_operation_id = _id("expert-validation-operation", "plan-reservation")
    plan_id = _id("expert-release-matrix-evaluation-plan", "plan")
    transition_id = _id("expert-validation-transition", "transition")
    state_id = _id("expert-candidate-validation-state", "state")
    attempt_id = _id("expert-validation-attempt", "attempt")
    candidate_id = _id("expert-candidate", "candidate")
    commit_id = _id("expert-candidate-commit", "candidate")
    scope_contract_id = _id("expert-scope-contract", "scope")
    policy_id = _id("expert-validation-policy", "policy")
    parent_id = (
        None
        if mode is ExpertReleaseMatrixMode.BOOTSTRAP
        else _id("expert-base-release", "parent")
    )
    plan_dependencies = tuple(
        sorted(
            (
                _id("expert-release-matrix-adapter-authority", "adapter"),
                _id("expert-release-matrix-provenance-binding", "provenance"),
                *case.evaluation_cell_ids,
            )
        )
    )
    dependencies = {
        plan_operation_id,
        plan_id,
        transition_id,
        state_id,
        attempt_id,
        candidate_id,
        commit_id,
        scope_contract_id,
        policy_id,
        *plan_dependencies,
        case.evaluation_case_id,
        *case.exact_dependency_ids,
    }
    if parent_id is not None:
        dependencies.add(parent_id)
    return TaskEvaluationRequest.mint(
        request_contract_version=TASK_EVALUATION_REQUEST_CONTRACT_VERSION,
        plan_reservation_operation_id=plan_operation_id,
        evaluation_plan_id=plan_id,
        mode=mode,
        authorization_transition_id=transition_id,
        authorization_state_id=state_id,
        validation_attempt_id=attempt_id,
        candidate_id=candidate_id,
        candidate_commit_record_id=commit_id,
        candidate_tree_hash=_digest("candidate-tree"),
        scope_contract_id=scope_contract_id,
        scope_id="ml_ai",
        parent_release_id=parent_id,
        parent_tree_hash=(None if parent_id is None else _digest("parent-tree")),
        validation_policy_id=policy_id,
        configuration_fingerprint=_digest("configuration"),
        release_matrix_evaluator_id="expert_release_matrix_evaluator",
        release_matrix_evaluator_role="expert_release_matrix_evaluator",
        release_matrix_evaluator_version="kapso.expert_release_matrix_evaluator.v1",
        plan_dependency_ids=plan_dependencies,
        cases=(case,),
        exact_dependency_ids=tuple(sorted(dependencies)),
    )


def _reservation(request: TaskEvaluationRequest) -> TaskEvaluationReservation:
    observation_id = _id(
        "task-evaluation-current-release-observation",
        "current",
    )
    dependencies = {
        request.request_id,
        request.plan_reservation_operation_id,
        request.evaluation_plan_id,
        request.authorization_transition_id,
        request.authorization_state_id,
        request.validation_attempt_id,
        request.candidate_id,
        request.scope_contract_id,
        observation_id,
    }
    if request.parent_release_id is not None:
        dependencies.add(request.parent_release_id)
    return TaskEvaluationReservation.mint(
        request_id=request.request_id,
        plan_reservation_operation_id=request.plan_reservation_operation_id,
        evaluation_plan_id=request.evaluation_plan_id,
        mode=request.mode,
        authorization_transition_id=request.authorization_transition_id,
        authorization_state_id=request.authorization_state_id,
        validation_attempt_id=request.validation_attempt_id,
        candidate_id=request.candidate_id,
        candidate_tree_hash=request.candidate_tree_hash,
        scope_contract_id=request.scope_contract_id,
        scope_id=request.scope_id,
        current_release_observation_id=observation_id,
        observed_current_release_id=request.parent_release_id,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )


def _request_for_reserved_plan(
    plan_reservation,
    settings,
    prepared_plan,
) -> TaskEvaluationRequest:
    stored = prepared_plan.stored_candidate
    candidate = VerifiedTaskEvaluationCandidate(
        manifest=stored.closure.manifest,
        commit_record=stored.commit_record,
        source_tree=stored.closure.candidate_tree,
        source_contents=stored.closure.candidate_contents,
    )
    packet = stored.closure.derivation.trigger_packet
    if packet.parent_release is None or packet.parent_tree_receipt is None:
        parent = None
    else:
        _released_packet, _materialized, parent_contents = released_workspace_fixture()
        parent = VerifiedTaskEvaluationParent(
            release_manifest=packet.parent_release,
            parent_tree_receipt=packet.parent_tree_receipt,
            source_contents=parent_contents,
        )
    return prepare_task_evaluation_request(
        plan_reservation=plan_reservation,
        settings=settings,
        stored_candidate=stored,
        candidate=candidate,
        parent=parent,
    ).request


def _request_with_cases(
    request: TaskEvaluationRequest,
    cases: tuple[TaskEvaluationCase, ...],
) -> TaskEvaluationRequest:
    values = request.to_dict()
    values.pop("request_id")
    values["cases"] = tuple(case.to_dict() for case in cases)
    dependencies = {
        request.plan_reservation_operation_id,
        request.evaluation_plan_id,
        request.authorization_transition_id,
        request.authorization_state_id,
        request.validation_attempt_id,
        request.candidate_id,
        request.candidate_commit_record_id,
        request.scope_contract_id,
        request.validation_policy_id,
        *request.plan_dependency_ids,
        *(case.evaluation_case_id for case in cases),
        *(
            dependency_id
            for case in cases
            for dependency_id in case.exact_dependency_ids
        ),
    }
    values["exact_dependency_ids"] = tuple(sorted(dependencies))
    return TaskEvaluationRequest.mint(**values)


def _case_with_updates(
    case: TaskEvaluationCase,
    **updates,
) -> TaskEvaluationCase:
    values = case.to_dict()
    values.pop("evaluation_case_id")
    values.update(updates)
    compute_binding = values["compute_binding"]
    legs = values["legs"]
    dependencies = {
        values["adapter_authority_id"],
        values["provenance_binding_id"],
        values["release_matrix_case_id"],
        values["task_context_binding_id"],
        values["independence_group_id"],
        *values["evaluation_cell_ids"],
        *values["evaluation_fingerprint_ids"],
        *values["starting_artifact_ids"],
        compute_binding["compute_binding_id"],
        *(leg["leg_id"] for leg in legs),
        *(
            dependency_id
            for leg in legs
            for dependency_id in leg["exact_dependency_ids"]
        ),
    }
    values["exact_dependency_ids"] = tuple(sorted(dependencies))
    return TaskEvaluationCase.mint(**values)


@pytest.mark.parametrize(
    "mode,expected_leg_kinds",
    (
        (
            ExpertReleaseMatrixMode.BOOTSTRAP,
            {TaskEvaluationLegKind.CANDIDATE},
        ),
        (
            ExpertReleaseMatrixMode.PARENT_COMPARISON,
            set(TaskEvaluationLegKind),
        ),
    ),
)
def test_request_round_trips_one_case_with_all_fingerprints_and_exact_legs(
    mode,
    expected_leg_kinds,
):
    request = _request(mode)
    case = request.cases[0]
    reservation = _reservation(request)

    assert len(case.evaluation_cell_ids) == len(case.evaluation_fingerprint_ids) == 2
    assert {leg.kind for leg in case.legs} == expected_leg_kinds
    assert set(case.compute_binding.leg_order) == expected_leg_kinds
    assert reservation.mode is mode
    assert TaskEvaluationRequest.from_json_bytes(request.to_json_bytes()) == request
    assert (
        TaskEvaluationReservation.from_json_bytes(reservation.to_json_bytes())
        == reservation
    )


def test_bootstrap_cannot_name_or_schedule_a_parent():
    bootstrap = _request(ExpertReleaseMatrixMode.BOOTSTRAP)
    parent = _request(ExpertReleaseMatrixMode.PARENT_COMPARISON)

    with pytest.raises(TaskEvaluationContractError, match="cannot name a parent"):
        _remint(
            bootstrap,
            parent_release_id=parent.parent_release_id,
            parent_tree_hash=parent.parent_tree_hash,
        )
    with pytest.raises(TaskEvaluationContractError, match="mode-specific legs"):
        _remint(
            bootstrap,
            cases=parent.cases,
        )
    with pytest.raises(TaskEvaluationContractError, match="dependency closure"):
        _remint(
            _reservation(bootstrap),
            current_release_observation_id=_id(
                "task-evaluation-current-release-observation",
                "substituted",
            ),
        )


def test_parent_mode_requires_both_semantic_legs():
    parent = _request(ExpertReleaseMatrixMode.PARENT_COMPARISON)
    candidate_case = _case(ExpertReleaseMatrixMode.BOOTSTRAP)

    with pytest.raises(TaskEvaluationContractError, match="requires a parent"):
        _remint(parent, parent_release_id=None, parent_tree_hash=None)
    with pytest.raises(TaskEvaluationContractError, match="mode-specific legs"):
        _remint(parent, cases=(candidate_case,))
    with pytest.raises(TaskEvaluationContractError, match="expert authority"):
        _remint(parent, parent_tree_hash=_digest("substituted-parent-tree"))


def test_request_legs_cannot_substitute_the_candidate_tree():
    bootstrap = _request(ExpertReleaseMatrixMode.BOOTSTRAP)

    with pytest.raises(TaskEvaluationContractError, match="expert authority"):
        _remint(
            bootstrap,
            candidate_tree_hash=_digest("substituted-candidate-tree"),
        )


def test_request_contract_version_and_provenance_roles_are_exact():
    request = _request(ExpertReleaseMatrixMode.PARENT_COMPARISON)
    with pytest.raises(TaskEvaluationContractError, match="version is unsupported"):
        _remint(
            request,
            request_contract_version="kapso.task_evaluation_request.v2",
        )

    foreign_case = _case_with_updates(
        request.cases[0],
        adapter_authority_id=_id(
            "expert-release-matrix-adapter-authority",
            "foreign-adapter",
        ),
    )
    duplicate_provenance_cases = tuple(
        sorted(
            (foreign_case, *request.cases),
            key=lambda case: case.canonical_key,
        )
    )
    with pytest.raises(TaskEvaluationContractError, match="unique provenances"):
        _request_with_cases(request, duplicate_provenance_cases)


def test_case_schedule_and_dependency_closures_fail_loud():
    case = _case(ExpertReleaseMatrixMode.PARENT_COMPARISON)
    compute = case.compute_binding

    with pytest.raises(TaskEvaluationContractError, match="no duplicates"):
        _remint(
            compute,
            leg_order=(
                TaskEvaluationLegKind.CANDIDATE,
                TaskEvaluationLegKind.CANDIDATE,
            ),
        )
    with pytest.raises(TaskEvaluationContractError, match="cover fingerprints"):
        _remint(case, evaluation_cell_ids=case.evaluation_cell_ids[:1])
    with pytest.raises(TaskEvaluationContractError, match="dependency closure"):
        _remint(case, exact_dependency_ids=case.exact_dependency_ids[:-1])
    request = _request(ExpertReleaseMatrixMode.PARENT_COMPARISON)
    with pytest.raises(TaskEvaluationContractError, match="dependency closure"):
        _remint(request, exact_dependency_ids=request.exact_dependency_ids[:-1])
    reservation = _reservation(request)
    with pytest.raises(TaskEvaluationContractError, match="dependency closure"):
        _remint(
            reservation,
            exact_dependency_ids=reservation.exact_dependency_ids[:-1],
        )


def test_plan_join_requires_exact_adapter_case_projection_and_evaluator_role(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
        add_active_case=True,
    )
    reserved = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared,
    ).reservation
    request = _request_for_reserved_plan(
        reserved,
        validation_store.settings,
        prepared,
    )

    joined = PlanJoinedTaskEvaluationRequest(
        request=request,
        plan_reservation=reserved,
        settings=validation_store.settings,
    )
    assert joined.request == request
    assert len(request.cases) == 2
    provenance_ids = tuple(case.provenance_binding_id for case in request.cases)
    expected_compute = derive_release_matrix_compute_bindings(
        settings=validation_store.settings,
        mode=request.mode,
        provenance_binding_ids=tuple(reversed(provenance_ids)),
    )
    assert {
        case.provenance_binding_id: case.compute_binding for case in request.cases
    } == expected_compute
    assert {case.compute_binding.leg_order for case in request.cases} == {
        (
            TaskEvaluationLegKind.PARENT_CONTROL,
            TaskEvaluationLegKind.CANDIDATE,
        ),
        (
            TaskEvaluationLegKind.CANDIDATE,
            TaskEvaluationLegKind.PARENT_CONTROL,
        ),
    }
    assert all(
        binding.execution_provider_settings_digest
        == tree_or_blob_digest(
            validation_store.settings.task_evaluation_provider.to_json_bytes()
        )
        for binding in expected_compute.values()
    )
    bootstrap_compute = derive_release_matrix_compute_bindings(
        settings=validation_store.settings,
        mode=ExpertReleaseMatrixMode.BOOTSTRAP,
        provenance_binding_ids=provenance_ids,
    )
    assert {binding.leg_order for binding in bootstrap_compute.values()} == {
        (TaskEvaluationLegKind.CANDIDATE,)
    }
    with pytest.raises(TaskEvaluationComputeError, match="exact matrix mode"):
        derive_release_matrix_compute_bindings(
            settings=validation_store.settings,
            mode="parent_comparison",
            provenance_binding_ids=provenance_ids,
        )

    incomplete_request = _request_with_cases(request, request.cases[:1])
    with pytest.raises(
        TaskEvaluationRequestPreparationError,
        match="provenance coverage is not exact",
    ):
        PlanJoinedTaskEvaluationRequest(
            request=incomplete_request,
            plan_reservation=reserved,
            settings=validation_store.settings,
        )

    substituted_case = _case_with_updates(
        request.cases[0],
        release_matrix_case_id=_id(
            "task-adapter-release-matrix-case",
            "substituted-signed-case",
        ),
    )
    substituted_case_request = _request_with_cases(
        request,
        tuple(
            sorted(
                (substituted_case, *request.cases[1:]),
                key=lambda case: case.canonical_key,
            )
        ),
    )
    with pytest.raises(
        TaskEvaluationRequestPreparationError,
        match="case differs from its reserved provenance",
    ):
        PlanJoinedTaskEvaluationRequest(
            request=substituted_case_request,
            plan_reservation=reserved,
            settings=validation_store.settings,
        )

    substituted_compute = _remint(
        request.cases[0].compute_binding,
        cpu_millicore_limit=(request.cases[0].compute_binding.cpu_millicore_limit + 1),
    )
    compute_substituted_case = _case_with_updates(
        request.cases[0],
        compute_binding=substituted_compute.to_dict(),
    )
    compute_substituted_request = _request_with_cases(
        request,
        tuple(
            sorted(
                (compute_substituted_case, *request.cases[1:]),
                key=lambda case: case.canonical_key,
            )
        ),
    )
    with pytest.raises(
        TaskEvaluationRequestPreparationError,
        match="case differs from its reserved provenance",
    ):
        PlanJoinedTaskEvaluationRequest(
            request=compute_substituted_request,
            plan_reservation=reserved,
            settings=validation_store.settings,
        )

    values = request.to_dict()
    values.pop("request_id")
    values["release_matrix_evaluator_role"] = "substituted_evaluator_role"
    substituted_role_request = TaskEvaluationRequest.mint(**values)
    with pytest.raises(
        TaskEvaluationRequestPreparationError,
        match="evaluator differs from configuration",
    ):
        PlanJoinedTaskEvaluationRequest(
            request=substituted_role_request,
            plan_reservation=reserved,
            settings=validation_store.settings,
        )


def test_plan_join_rejects_a_request_from_another_reserved_plan(
    tmp_path,
    monkeypatch,
):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_store, first_snapshot, first_prepared = _release_matrix_fixture(
        first_root,
        monkeypatch,
        add_active_case=True,
    )
    second_store, second_snapshot, second_prepared = _release_matrix_fixture(
        second_root,
        monkeypatch,
    )
    first_reservation = first_store.reserve_release_matrix_plan(
        expected_transition_id=first_snapshot.transition.transition_id,
        prepared_plan=first_prepared,
    ).reservation
    second_reservation = second_store.reserve_release_matrix_plan(
        expected_transition_id=second_snapshot.transition.transition_id,
        prepared_plan=second_prepared,
    ).reservation
    first_request = _request_for_reserved_plan(
        first_reservation,
        first_store.settings,
        first_prepared,
    )

    with pytest.raises(
        TaskEvaluationRequestPreparationError,
        match="reserved plan subjects",
    ):
        PlanJoinedTaskEvaluationRequest(
            request=first_request,
            plan_reservation=second_reservation,
            settings=second_store.settings,
        )


def test_request_derivation_enforces_parent_receipt_mode(
    tmp_path,
    monkeypatch,
):
    parent_root = tmp_path / "parent"
    bootstrap_root = tmp_path / "bootstrap"
    parent_root.mkdir()
    bootstrap_root.mkdir()
    parent_store, parent_snapshot, parent_prepared = _release_matrix_fixture(
        parent_root,
        monkeypatch,
    )
    bootstrap_store, bootstrap_snapshot, bootstrap_prepared, _adapter_provider = (
        _bootstrap_release_matrix_fixture(bootstrap_root, monkeypatch)
    )
    parent_reservation = parent_store.reserve_release_matrix_plan(
        expected_transition_id=parent_snapshot.transition.transition_id,
        prepared_plan=parent_prepared,
    ).reservation
    bootstrap_reservation = bootstrap_store.reserve_release_matrix_plan(
        expected_transition_id=bootstrap_snapshot.transition.transition_id,
        prepared_plan=bootstrap_prepared,
    ).reservation

    with pytest.raises(
        TaskEvaluationRequestPreparationError,
        match="parent authority differs from matrix mode",
    ):
        parent_candidate = VerifiedTaskEvaluationCandidate(
            manifest=parent_prepared.stored_candidate.closure.manifest,
            commit_record=parent_prepared.stored_candidate.commit_record,
            source_tree=parent_prepared.stored_candidate.closure.candidate_tree,
            source_contents=(
                parent_prepared.stored_candidate.closure.candidate_contents
            ),
        )
        prepare_task_evaluation_request(
            plan_reservation=parent_reservation,
            settings=parent_store.settings,
            stored_candidate=parent_prepared.stored_candidate,
            candidate=parent_candidate,
            parent=None,
        )
    bootstrap_candidate = VerifiedTaskEvaluationCandidate(
        manifest=bootstrap_prepared.stored_candidate.closure.manifest,
        commit_record=bootstrap_prepared.stored_candidate.commit_record,
        source_tree=bootstrap_prepared.stored_candidate.closure.candidate_tree,
        source_contents=bootstrap_prepared.stored_candidate.closure.candidate_contents,
    )
    _released_packet, _materialized, parent_contents = released_workspace_fixture()
    parent_packet = parent_prepared.stored_candidate.closure.derivation.trigger_packet
    assert parent_packet.parent_release is not None
    assert parent_packet.parent_tree_receipt is not None
    exact_parent = VerifiedTaskEvaluationParent(
        release_manifest=parent_packet.parent_release,
        parent_tree_receipt=parent_packet.parent_tree_receipt,
        source_contents=parent_contents,
    )
    parent_receipt = exact_parent.parent_tree_receipt
    substituted_receipt = ExpertParentTreeReceipt.mint(
        release_id=parent_receipt.release_id,
        cache_verification_receipt=parent_receipt.cache_verification_receipt,
        source_extraction_receipt=parent_receipt.source_extraction_receipt,
        parent_tree_hash=parent_receipt.parent_tree_hash,
        repository_map_id=parent_receipt.repository_map_id,
        module_contract_ids=parent_receipt.module_contract_ids,
        materializer_version="substituted.materializer.v1",
    )
    substituted_parent = VerifiedTaskEvaluationParent(
        release_manifest=parent_packet.parent_release,
        parent_tree_receipt=substituted_receipt,
        source_contents=parent_contents,
    )
    substituted_packet = ExpertTriggerEvidencePacket.mint(
        knowledge_snapshot_manifest=parent_packet.knowledge_snapshot_manifest,
        knowledge_record_closure_digest=(parent_packet.knowledge_record_closure_digest),
        configuration_fingerprint=parent_packet.configuration_fingerprint,
        scope_contract=parent_packet.scope_contract,
        parent_scope_contract=parent_packet.parent_scope_contract,
        parent_release=parent_packet.parent_release,
        parent_tree_receipt=substituted_receipt,
        parent_tree_hash=parent_packet.parent_tree_hash,
        repository_map=parent_packet.repository_map,
        module_contracts=parent_packet.module_contracts,
        episodes=parent_packet.episodes,
        claims=parent_packet.claims,
        trigger_observations=parent_packet.trigger_observations,
        active_task_bindings=parent_packet.active_task_bindings,
        proof_reference_ids=parent_packet.proof_reference_ids,
    )
    substituted_stored_candidate = replace(
        parent_prepared.stored_candidate,
        closure=replace(
            parent_prepared.stored_candidate.closure,
            trigger_packet=substituted_packet,
        ),
    )
    with pytest.raises(
        TaskEvaluationRequestPreparationError,
        match="candidate differs from reserved plan authority",
    ):
        prepare_task_evaluation_request(
            plan_reservation=parent_reservation,
            settings=parent_store.settings,
            stored_candidate=substituted_stored_candidate,
            candidate=parent_candidate,
            parent=substituted_parent,
        )
    with pytest.raises(
        TaskEvaluationRequestPreparationError,
        match="parent differs from reserved plan authority",
    ):
        prepare_task_evaluation_request(
            plan_reservation=parent_reservation,
            settings=parent_store.settings,
            stored_candidate=parent_prepared.stored_candidate,
            candidate=parent_candidate,
            parent=substituted_parent,
        )
    parent_request = prepare_task_evaluation_request(
        plan_reservation=parent_reservation,
        settings=parent_store.settings,
        stored_candidate=parent_prepared.stored_candidate,
        candidate=parent_candidate,
        parent=exact_parent,
    ).request
    assert {
        leg.expert_source_receipt_id
        for case in parent_request.cases
        for leg in case.legs
        if leg.kind is TaskEvaluationLegKind.PARENT_CONTROL
    } == {parent_receipt.parent_tree_receipt_id}
    with pytest.raises(
        TaskEvaluationRequestPreparationError,
        match="parent authority differs from matrix mode",
    ):
        prepare_task_evaluation_request(
            plan_reservation=bootstrap_reservation,
            settings=bootstrap_store.settings,
            stored_candidate=bootstrap_prepared.stored_candidate,
            candidate=bootstrap_candidate,
            parent=exact_parent,
        )
    bootstrap_request = prepare_task_evaluation_request(
        plan_reservation=bootstrap_reservation,
        settings=bootstrap_store.settings,
        stored_candidate=bootstrap_prepared.stored_candidate,
        candidate=bootstrap_candidate,
        parent=None,
    ).request
    assert all(
        tuple(leg.kind for leg in case.legs) == (TaskEvaluationLegKind.CANDIDATE,)
        for case in bootstrap_request.cases
    )
