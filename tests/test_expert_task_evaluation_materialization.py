from collections.abc import Mapping
from dataclasses import replace

import pytest

from kapso.cross_run.canonical import source_tree_digest, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ObjectiveDirection,
    SourceFileDescriptor,
    TaskAdapterManifest,
    TaskAdapterReleaseMatrixStartingArtifact,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    TaskEvaluationMaterializationError,
    VerifiedTaskEvaluationAdapterRuntime,
    VerifiedTaskEvaluationStartingArtifact,
    materialize_task_evaluation_starting_artifacts,
)
from kapso.cross_run.task_adapters import TaskAdapterVerificationReceipt
from task_adapter_matrix_fixtures import task_adapter_release_matrix_case
from test_cross_run_contracts import (
    build_records,
    verified_test_task_adapter,
)
from test_expert_source_replay_request import _prepared, _request_fixture


class _ChangingMapping(Mapping):
    def __init__(self, path: str, first_payload: bytes, later_payload: bytes):
        self.path = path
        self.first_payload = first_payload
        self.later_payload = later_payload
        self.read_count = 0

    def __getitem__(self, key):
        if key != self.path:
            raise KeyError(key)
        self.read_count += 1
        return self.first_payload if self.read_count == 1 else self.later_payload

    def __iter__(self):
        return iter((self.path,))

    def __len__(self):
        return 1


def _starting_artifact(label: str, payload: bytes):
    descriptor = SourceFileDescriptor(
        relative_path="fixture.json",
        digest=tree_or_blob_digest(payload),
        mode="100644",
        size=len(payload),
    )
    return TaskAdapterReleaseMatrixStartingArtifact.mint(
        starting_artifact_ref=f"artifact/{label}",
        mount_path=label,
        package_source_root=f"release_matrix_assets/{label}",
        materialized_tree_hash=source_tree_digest(
            {
                descriptor.relative_path: (
                    descriptor.digest,
                    descriptor.mode,
                    descriptor.size,
                )
            }
        ),
        source_files=(descriptor,),
    )


def _adapter_with_two_cases():
    first_payload = b'{"case": "first"}\n'
    second_payload = b'{"case": "second"}\n'
    first_artifact = _starting_artifact("first", first_payload)
    second_artifact = _starting_artifact("second", second_payload)
    source_contents = {
        "adapter.py": b"ADAPTER_ID = 'posttrain'\n",
        "requirements.lock": b"python==3.11.9\n",
        "release_matrix_assets/first/fixture.json": first_payload,
        "release_matrix_assets/second/fixture.json": second_payload,
    }
    records = build_records(
        task_adapter_source_contents=source_contents,
        task_adapter_release_matrix_starting_artifacts=(
            first_artifact,
            second_artifact,
        ),
    )
    base_manifest = next(
        record for record in records if type(record) is TaskAdapterManifest
    )
    base_case = base_manifest.release_matrix_cases[0]
    comparison_binding = base_manifest.task_evaluator.metric_comparison_bindings[0]
    cases = tuple(
        sorted(
            (
                task_adapter_release_matrix_case(
                    scope_contract_id=base_manifest.scope_contract_id,
                    scope_id=base_case.task_context_binding.scope_id,
                    task_family_id=base_manifest.task_family_id,
                    task_adapter_id=base_manifest.task_adapter_id,
                    evaluator_fingerprint=comparison_binding.evaluator_fingerprint,
                    metric_directions=(
                        (
                            comparison_binding.metric_name,
                            ObjectiveDirection.MAXIMIZE,
                        ),
                    ),
                    transfer_dimensions={
                        "dataset_family": "instruction",
                        "runtime_family": "pytorch",
                    },
                    label=label,
                    starting_artifacts=(artifact,),
                )
                for label, artifact in (
                    ("first", first_artifact),
                    ("second", second_artifact),
                )
            ),
            key=lambda case: case.release_matrix_case_id,
        )
    )
    manifest = TaskAdapterManifest.mint(
        task_adapter_id=base_manifest.task_adapter_id,
        scope_contract_id=base_manifest.scope_contract_id,
        task_family_id=base_manifest.task_family_id,
        publisher_attestation=base_manifest.publisher_attestation,
        task_evaluator=base_manifest.task_evaluator,
        context_binding=base_manifest.context_binding,
        release_matrix_cases=cases,
        source_tree_ref=base_manifest.source_tree_ref,
        tree_hash=base_manifest.tree_hash,
        runtime=base_manifest.runtime,
        sanitation_report_id=base_manifest.sanitation_report_id,
        validation_refs=base_manifest.validation_refs,
    )
    return (
        verified_test_task_adapter(manifest, source_contents=source_contents),
        first_payload,
        second_payload,
    )


def test_adapter_runtime_excludes_every_release_matrix_fixture_byte():
    adapter, first_payload, second_payload = _adapter_with_two_cases()

    runtime = VerifiedTaskEvaluationAdapterRuntime.from_verified_adapter(adapter)

    assert runtime.source_files == adapter.evaluation_runtime_source_files
    assert runtime.source_contents == adapter.evaluation_runtime_source_contents
    assert all(
        not descriptor.relative_path.startswith("release_matrix_assets/")
        for descriptor in runtime.source_files
    )
    assert first_payload not in runtime.source_contents.values()
    assert second_payload not in runtime.source_contents.values()
    with pytest.raises(TypeError):
        runtime.source_contents["adapter.py"] = b"substituted"


def test_adapter_runtime_rejects_full_package_or_non_adapter_input():
    adapter, _first_payload, _second_payload = _adapter_with_two_cases()
    runtime = VerifiedTaskEvaluationAdapterRuntime.from_verified_adapter(adapter)

    with pytest.raises(
        TaskEvaluationMaterializationError,
        match="differs from its verified package",
    ):
        replace(
            runtime,
            source_files=adapter.source_extraction_receipt.source_tree_files,
            source_contents=adapter.source_contents,
        )
    with pytest.raises(
        TaskEvaluationMaterializationError,
        match="requires an exact verified package",
    ):
        VerifiedTaskEvaluationAdapterRuntime.from_verified_adapter(object())


def test_adapter_runtime_rejects_substituted_receipt_proof_closure():
    adapter, _first_payload, _second_payload = _adapter_with_two_cases()
    receipt = adapter.verification_receipt
    substituted_receipt = TaskAdapterVerificationReceipt.mint(
        task_adapter_manifest_id=receipt.task_adapter_manifest_id,
        full_manifest_digest=receipt.full_manifest_digest,
        publisher_attestation_digest=receipt.publisher_attestation_digest,
        source_extraction_receipt_id=receipt.source_extraction_receipt_id,
        source_archive_ref=receipt.source_archive_ref,
        source_archive_digest=receipt.source_archive_digest,
        source_tree_hash=receipt.source_tree_hash,
        proof_object_digests={"foreign_proof": tree_or_blob_digest(b"foreign proof")},
        publisher_verification_digest=receipt.publisher_verification_digest,
        verifier_id=receipt.verifier_id,
        verifier_version=receipt.verifier_version,
    )

    with pytest.raises(
        TaskEvaluationMaterializationError,
        match="differs from its verified package",
    ):
        VerifiedTaskEvaluationAdapterRuntime(
            manifest=adapter.manifest,
            verification_receipt=substituted_receipt,
            source_extraction_receipt=adapter.source_extraction_receipt,
            source_files=adapter.evaluation_runtime_source_files,
            source_contents=adapter.evaluation_runtime_source_contents,
        )


def test_starting_artifact_selection_materializes_only_the_signed_case():
    adapter, first_payload, second_payload = _adapter_with_two_cases()
    selected_case = next(
        case
        for case in adapter.manifest.release_matrix_cases
        if case.starting_artifacts[0].starting_artifact_ref == "artifact/first"
    )

    artifacts = materialize_task_evaluation_starting_artifacts(
        adapter=adapter,
        signed_case=selected_case,
    )

    assert (
        tuple(item.artifact for item in artifacts) == selected_case.starting_artifacts
    )
    assert tuple(artifacts[0].source_contents.values()) == (first_payload,)
    assert second_payload not in artifacts[0].source_contents.values()


def test_starting_artifacts_reject_foreign_cases_and_substituted_bytes():
    adapter, _first_payload, _second_payload = _adapter_with_two_cases()
    selected_case = adapter.manifest.release_matrix_cases[0]
    foreign_case = task_adapter_release_matrix_case(
        scope_contract_id=adapter.manifest.scope_contract_id,
        scope_id=selected_case.task_context_binding.scope_id,
        task_family_id=adapter.manifest.task_family_id,
        task_adapter_id=adapter.manifest.task_adapter_id,
        evaluator_fingerprint=(
            adapter.manifest.task_evaluator.metric_comparison_bindings[
                0
            ].evaluator_fingerprint
        ),
        metric_directions=(("quality", ObjectiveDirection.MAXIMIZE),),
        transfer_dimensions={
            "dataset_family": "instruction",
            "runtime_family": "pytorch",
        },
        label="foreign",
    )

    with pytest.raises(
        TaskEvaluationMaterializationError,
        match="differs from its verified package",
    ):
        materialize_task_evaluation_starting_artifacts(
            adapter=adapter,
            signed_case=foreign_case,
        )
    with pytest.raises(
        TaskEvaluationMaterializationError,
        match="differ from their descriptor",
    ):
        VerifiedTaskEvaluationStartingArtifact(
            artifact=selected_case.starting_artifacts[0],
            source_contents={"fixture.json": b"substituted"},
        )


def test_source_byte_authority_snapshots_a_stateful_mapping_once():
    adapter, _first_payload, _second_payload = _adapter_with_two_cases()
    selected_artifact = adapter.manifest.release_matrix_cases[0].starting_artifacts[0]
    descriptor = selected_artifact.source_files[0]
    expected_payload = adapter.source_contents[
        f"{selected_artifact.package_source_root}/{descriptor.relative_path}"
    ]
    changing_contents = _ChangingMapping(
        descriptor.relative_path,
        expected_payload,
        b"substituted after validation",
    )

    verified = VerifiedTaskEvaluationStartingArtifact(
        artifact=selected_artifact,
        source_contents=changing_contents,
    )

    assert verified.source_contents[descriptor.relative_path] == expected_payload
    assert changing_contents.read_count == 1


def test_candidate_and_parent_authorities_reject_substituted_source_bytes(tmp_path):
    prepared = _prepared(_request_fixture(tmp_path))
    candidate_contents = dict(prepared.candidate.source_contents)
    candidate_path = next(iter(candidate_contents))
    candidate_contents[candidate_path] += b"substituted"
    parent_contents = dict(prepared.parent.source_contents)
    parent_path = next(iter(parent_contents))
    parent_contents[parent_path] += b"substituted"

    with pytest.raises(
        TaskEvaluationMaterializationError,
        match="candidate task-evaluation source bytes differ",
    ):
        replace(prepared.candidate, source_contents=candidate_contents)
    with pytest.raises(
        TaskEvaluationMaterializationError,
        match="parent task-evaluation source bytes differ",
    ):
        replace(prepared.parent, source_contents=parent_contents)
