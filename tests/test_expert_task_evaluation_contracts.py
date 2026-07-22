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
from kapso.cross_run.expert.task_evaluation_request import (
    PlanJoinedTaskEvaluationRequest,
    TaskEvaluationRequestPreparationError,
)
from test_expert_release_matrix_reservation import _release_matrix_fixture


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
    scope_id = _id("expert-scope-contract", "scope")
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
        scope_id,
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
        scope_contract_id=scope_id,
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
    dependencies = {
        request.request_id,
        request.plan_reservation_operation_id,
        request.evaluation_plan_id,
        request.authorization_transition_id,
        request.authorization_state_id,
        request.validation_attempt_id,
        request.candidate_id,
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
        observed_parent_release_id=request.parent_release_id,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )


def _request_for_reserved_plan(
    plan_reservation,
    settings,
) -> TaskEvaluationRequest:
    plan = plan_reservation.evaluation_plan
    snapshot = plan_reservation.snapshot
    evaluator = next(
        evaluator
        for evaluator in settings.policy.evaluators
        if evaluator.stage is ExpertValidationStage.RELEASE_MATRIX
    )
    candidate_leg = TaskEvaluationExpertLeg.mint(
        kind=TaskEvaluationLegKind.CANDIDATE,
        expert_artifact_id=plan.candidate_id,
        expert_source_receipt_id=plan.candidate_commit_record_id,
        expert_tree_hash=plan.candidate_tree_hash,
        exact_dependency_ids=tuple(
            sorted((plan.candidate_id, plan.candidate_commit_record_id))
        ),
    )
    if plan.parent_release_id is None:
        legs = (candidate_leg,)
        leg_order = (TaskEvaluationLegKind.CANDIDATE,)
    else:
        parent_receipt_id = _id(
            "expert-parent-tree-receipt",
            plan.parent_release_id,
        )
        parent_leg = TaskEvaluationExpertLeg.mint(
            kind=TaskEvaluationLegKind.PARENT_CONTROL,
            expert_artifact_id=plan.parent_release_id,
            expert_source_receipt_id=parent_receipt_id,
            expert_tree_hash=plan.parent_tree_hash,
            exact_dependency_ids=tuple(
                sorted((plan.parent_release_id, parent_receipt_id))
            ),
        )
        legs = tuple(sorted((parent_leg, candidate_leg), key=lambda leg: leg.leg_id))
        leg_order = (
            TaskEvaluationLegKind.PARENT_CONTROL,
            TaskEvaluationLegKind.CANDIDATE,
        )
    compute = _compute(leg_order)
    cases = []
    for provenance in plan.provenance_bindings:
        if provenance.adapter_case is None:
            continue
        planned_cells = tuple(
            cell
            for cell in plan.evaluation_cells
            if cell.provenance_binding_id == provenance.provenance_binding_id
        )
        case_dependencies = {
            provenance.adapter_authority_id,
            provenance.provenance_binding_id,
            provenance.adapter_case.release_matrix_case_id,
            provenance.task_context_binding.task_context_binding_id,
            provenance.adapter_case.independence_group.independence_group_id,
            *(cell.evaluation_cell_id for cell in planned_cells),
            *provenance.evaluation_fingerprint_ids,
            *provenance.starting_artifact_ids,
            compute.compute_binding_id,
            *(leg.leg_id for leg in legs),
            *(
                dependency_id
                for leg in legs
                for dependency_id in leg.exact_dependency_ids
            ),
        }
        cases.append(
            TaskEvaluationCase.mint(
                adapter_authority_id=provenance.adapter_authority_id,
                provenance_binding_id=provenance.provenance_binding_id,
                release_matrix_case_id=(provenance.adapter_case.release_matrix_case_id),
                task_context_binding_id=(
                    provenance.task_context_binding.task_context_binding_id
                ),
                independence_group_id=(
                    provenance.adapter_case.independence_group.independence_group_id
                ),
                evaluation_cell_ids=tuple(
                    sorted(cell.evaluation_cell_id for cell in planned_cells)
                ),
                evaluation_fingerprint_ids=provenance.evaluation_fingerprint_ids,
                starting_artifact_ids=provenance.starting_artifact_ids,
                compute_binding=compute,
                legs=legs,
                exact_dependency_ids=tuple(sorted(case_dependencies)),
            )
        )
    canonical_cases = tuple(sorted(cases, key=lambda case: case.canonical_key))
    request_dependencies = {
        plan_reservation.operation.operation_id,
        plan.evaluation_plan_id,
        snapshot.transition.transition_id,
        snapshot.state.validation_state_id,
        plan.validation_attempt_id,
        plan.candidate_id,
        plan.candidate_commit_record_id,
        plan.scope_contract_id,
        plan.validation_policy_id,
        *plan.exact_dependency_ids,
        *(case.evaluation_case_id for case in canonical_cases),
        *(
            dependency_id
            for case in canonical_cases
            for dependency_id in case.exact_dependency_ids
        ),
    }
    if plan.parent_release_id is not None:
        request_dependencies.add(plan.parent_release_id)
    return TaskEvaluationRequest.mint(
        request_contract_version=TASK_EVALUATION_REQUEST_CONTRACT_VERSION,
        plan_reservation_operation_id=plan_reservation.operation.operation_id,
        evaluation_plan_id=plan.evaluation_plan_id,
        mode=plan.mode,
        authorization_transition_id=snapshot.transition.transition_id,
        authorization_state_id=snapshot.state.validation_state_id,
        validation_attempt_id=plan.validation_attempt_id,
        candidate_id=plan.candidate_id,
        candidate_commit_record_id=plan.candidate_commit_record_id,
        candidate_tree_hash=plan.candidate_tree_hash,
        scope_contract_id=plan.scope_contract_id,
        parent_release_id=plan.parent_release_id,
        parent_tree_hash=plan.parent_tree_hash,
        validation_policy_id=plan.validation_policy_id,
        configuration_fingerprint=plan.configuration_fingerprint,
        release_matrix_evaluator_id=evaluator.evaluator_id,
        release_matrix_evaluator_role=evaluator.evaluator_role,
        release_matrix_evaluator_version=evaluator.evaluator_version,
        plan_dependency_ids=plan.exact_dependency_ids,
        cases=canonical_cases,
        exact_dependency_ids=tuple(sorted(request_dependencies)),
    )


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
    with pytest.raises(TaskEvaluationContractError, match="parent differs"):
        _remint(
            _reservation(bootstrap),
            observed_parent_release_id=parent.parent_release_id,
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
    )

    joined = PlanJoinedTaskEvaluationRequest(
        request=request,
        plan_reservation=reserved,
        settings=validation_store.settings,
    )
    assert joined.request == request
    assert len(request.cases) == 2

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
