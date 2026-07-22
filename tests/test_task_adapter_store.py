import io
import json
import stat
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
import zstandard

import kapso.cross_run.task_adapter_store as task_adapter_storage
from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    parse_json_bytes,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ContractValidationError,
    ObjectiveDirection,
    SourceFileDescriptor,
    TaskAdapterContextBinding,
    TaskAdapterManifest,
    TaskAdapterReleaseMatrixStartingArtifact,
    TaskAdapterRuntimeContract,
    TaskEvaluatorBinding,
    TaskEvaluatorMetricComparisonBinding,
)
from kapso.cross_run.expert.validation import ExpertCandidateEligibilityEvaluator
from kapso.cross_run.settings import (
    CrossRunSettings,
    TaskAdapterAuthorityTrustSettings,
)
from kapso.cross_run.task_adapter_store import (
    TaskAdapterActivationConflict,
    TaskAdapterAuthorityRegistry,
    TaskAdapterPackageStore,
    TaskAdapterStoreError,
)
from kapso.cross_run.task_adapters import (
    TaskAdapterPackage,
    task_adapter_materialization_usage,
)
from test_expert_candidate_store import candidate_store
from test_expert_candidates import bootstrap_candidate_closure
from test_expert_validation import _CurrentReleaseProvider
from task_adapter_matrix_fixtures import task_adapter_release_matrix_case

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
RUNTIME_LOCK = b"python==3.11.9\n"


def _content_id(label: str) -> str:
    return content_id("test-task-adapter-store", {"label": label})


def _starting_artifact(
    payload: bytes,
    *,
    label: str = "seed",
    package_source_root: str | None = None,
) -> TaskAdapterReleaseMatrixStartingArtifact:
    descriptor = SourceFileDescriptor(
        relative_path="fixture.json",
        digest=tree_or_blob_digest(payload),
        mode="100644",
        size=len(payload),
    )
    return TaskAdapterReleaseMatrixStartingArtifact.mint(
        starting_artifact_ref=f"artifact/{label}",
        mount_path=f"inputs/{label}",
        package_source_root=(
            f"release_matrix_assets/{label}"
            if package_source_root is None
            else package_source_root
        ),
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


def _source_descriptors(
    source: bytes,
    *,
    executable: bool = True,
    extra_files: dict[str, bytes] | None = None,
) -> tuple[SourceFileDescriptor, ...]:
    selected_extra_files = {} if extra_files is None else extra_files
    descriptors = (
        SourceFileDescriptor(
            relative_path="adapter.py",
            digest=tree_or_blob_digest(source),
            mode="100755" if executable else "100644",
            size=len(source),
        ),
        SourceFileDescriptor(
            relative_path="requirements.lock",
            digest=tree_or_blob_digest(RUNTIME_LOCK),
            mode="100644",
            size=len(RUNTIME_LOCK),
        ),
        *(
            SourceFileDescriptor(
                relative_path=path,
                digest=tree_or_blob_digest(payload),
                mode="100644",
                size=len(payload),
            )
            for path, payload in selected_extra_files.items()
        ),
    )
    return tuple(sorted(descriptors, key=lambda descriptor: descriptor.relative_path))


def _source_tree_hash(
    source: bytes,
    *,
    executable: bool = True,
    extra_files: dict[str, bytes] | None = None,
) -> str:
    descriptors = _source_descriptors(
        source,
        executable=executable,
        extra_files=extra_files,
    )
    return source_tree_digest(
        {
            descriptor.relative_path: (
                descriptor.digest,
                descriptor.mode,
                descriptor.size,
            )
            for descriptor in descriptors
        }
    )


def _archive(
    tmp_path: Path,
    source: bytes,
    *,
    executable: bool = True,
    extra_files: dict[str, bytes] | None = None,
) -> bytes:
    tar_path = tmp_path / "adapter.tar"
    with tarfile.open(tar_path, "w", format=tarfile.USTAR_FORMAT) as package:
        member = tarfile.TarInfo("adapter.py")
        member.size = len(source)
        member.mode = 0o755 if executable else 0o644
        member.uid = 0
        member.gid = 0
        member.mtime = 0
        package.addfile(member, io.BytesIO(source))
        lock_member = tarfile.TarInfo("requirements.lock")
        lock_member.size = len(RUNTIME_LOCK)
        lock_member.mode = 0o644
        lock_member.uid = 0
        lock_member.gid = 0
        lock_member.mtime = 0
        package.addfile(lock_member, io.BytesIO(RUNTIME_LOCK))
        for path, payload in sorted(
            ({} if extra_files is None else extra_files).items()
        ):
            extra_member = tarfile.TarInfo(path)
            extra_member.size = len(payload)
            extra_member.mode = 0o644
            extra_member.uid = 0
            extra_member.gid = 0
            extra_member.mtime = 0
            package.addfile(extra_member, io.BytesIO(payload))
    compressor = zstandard.ZstdCompressor(
        level=3,
        write_checksum=True,
        write_content_size=True,
    )
    return compressor.compress(tar_path.read_bytes())


def _manifest(
    source: bytes,
    publisher: str = "publisher-a",
    *,
    scope_contract_id: str | None = None,
    task_family_id: str = "posttrain",
    task_adapter_id: str = "posttrain_adapter",
    executable: bool = True,
    extra_files: dict[str, bytes] | None = None,
    starting_artifacts: tuple[TaskAdapterReleaseMatrixStartingArtifact, ...] = (),
    transfer_dimensions: dict[str, object] | None = None,
) -> TaskAdapterManifest:
    selected_scope_contract_id = (
        _content_id("scope") if scope_contract_id is None else scope_contract_id
    )
    evaluator_fingerprint = tree_or_blob_digest(b"source-evaluator")
    return TaskAdapterManifest.mint(
        task_adapter_id=task_adapter_id,
        scope_contract_id=selected_scope_contract_id,
        task_family_id=task_family_id,
        publisher_attestation={"publisher": publisher, "signature": "verified"},
        task_evaluator=TaskEvaluatorBinding(
            protocol_version="kapso.task_evaluator.v1",
            executable_path="adapter.py",
            supported_evaluator_fingerprints=(evaluator_fingerprint,),
            metric_comparison_bindings=(
                TaskEvaluatorMetricComparisonBinding(
                    evaluator_fingerprint=evaluator_fingerprint,
                    metric_name="accuracy",
                    objective_direction=ObjectiveDirection.MAXIMIZE,
                    comparison_dimension_id="quality",
                    comparison_scale=1.0,
                ),
            ),
        ),
        context_binding=TaskAdapterContextBinding(consumed_dimension_ids=()),
        release_matrix_cases=(
            task_adapter_release_matrix_case(
                scope_contract_id=selected_scope_contract_id,
                scope_id="ml_ai",
                task_family_id=task_family_id,
                task_adapter_id=task_adapter_id,
                evaluator_fingerprint=evaluator_fingerprint,
                metric_directions=(("accuracy", ObjectiveDirection.MAXIMIZE),),
                transfer_dimensions=(
                    {} if transfer_dimensions is None else transfer_dimensions
                ),
                label=f"{task_family_id}:{task_adapter_id}",
                starting_artifacts=starting_artifacts,
            ),
        ),
        source_tree_ref="task-adapter.tar.zst",
        tree_hash=_source_tree_hash(
            source,
            executable=executable,
            extra_files=extra_files,
        ),
        runtime=TaskAdapterRuntimeContract(
            runtime_protocol_version="kapso.task_adapter_runtime.v1",
            image_repository="registry.example/kapso/test-runtime",
            image_manifest_digest=tree_or_blob_digest(b"test-runtime-image"),
            image_config_digest=tree_or_blob_digest(b"test-runtime-config"),
            dependency_lock_path="requirements.lock",
            dependency_lock_digest=tree_or_blob_digest(RUNTIME_LOCK),
            operating_system="linux",
            architecture="amd64",
            architecture_variant=None,
            environment={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
        ),
        sanitation_report_id=_content_id("sanitation"),
        validation_refs=("validation.adapter_smoke",),
    )


def _remint_manifest(
    manifest: TaskAdapterManifest,
    **changes,
) -> TaskAdapterManifest:
    values = manifest.to_dict()
    values.pop("task_adapter_manifest_id")
    values.update(changes)
    return TaskAdapterManifest.mint(**values)


def _proof_objects(manifest: TaskAdapterManifest) -> dict[str, bytes]:
    return {
        proof_ref: canonical_json_bytes(
            {
                "manifest_id": manifest.task_adapter_manifest_id,
                "outcome": "passed",
                "proof_ref": proof_ref,
                "tree_hash": manifest.tree_hash,
            }
        )
        for proof_ref in {manifest.sanitation_report_id, *manifest.validation_refs}
    }


def _publisher_verification(
    manifest: TaskAdapterManifest,
    source_archive: bytes,
    proof_objects: dict[str, bytes],
) -> bytes:
    return canonical_json_bytes(
        {
            "archive_digest": tree_or_blob_digest(source_archive),
            "full_manifest_digest": tree_or_blob_digest(manifest.to_json_bytes()),
            "manifest_id": manifest.task_adapter_manifest_id,
            "proof_digests": {
                proof_ref: tree_or_blob_digest(payload)
                for proof_ref, payload in proof_objects.items()
            },
            "publisher_attestation": manifest.publisher_attestation,
            "tree_hash": manifest.tree_hash,
        }
    )


def _package(
    tmp_path: Path,
    manifest: TaskAdapterManifest,
    *,
    executable: bool = True,
    extra_files: dict[str, bytes] | None = None,
) -> TaskAdapterPackage:
    source = b"def evaluate(value):\n    return value + 1\n"
    assert manifest.tree_hash == _source_tree_hash(
        source,
        executable=executable,
        extra_files=extra_files,
    )
    source_archive = _archive(
        tmp_path,
        source,
        executable=executable,
        extra_files=extra_files,
    )
    proofs = _proof_objects(manifest)
    return TaskAdapterPackage(
        manifest=manifest,
        source_archive=source_archive,
        proof_objects=proofs,
        publisher_verification=_publisher_verification(
            manifest,
            source_archive,
            proofs,
        ),
    )


class _Authority:
    authority_id = "test_task_adapter_authority"
    authority_version = "test.task_adapter_authority.v1"

    def verify_package(
        self,
        *,
        manifest,
        source_extraction_receipt,
        proof_objects,
        publisher_verification,
    ):
        expected_proof_refs = {
            manifest.sanitation_report_id,
            *manifest.validation_refs,
        }
        if set(proof_objects) != expected_proof_refs:
            raise TaskAdapterStoreError("authority rejected proof closure")
        for proof_ref, payload in proof_objects.items():
            proof = parse_json_bytes(payload)
            if proof != {
                "manifest_id": manifest.task_adapter_manifest_id,
                "outcome": "passed",
                "proof_ref": proof_ref,
                "tree_hash": manifest.tree_hash,
            }:
                raise TaskAdapterStoreError("authority rejected task adapter proof")
        verification = parse_json_bytes(publisher_verification)
        expected = {
            "archive_digest": source_extraction_receipt.source_archive_digest,
            "full_manifest_digest": tree_or_blob_digest(manifest.to_json_bytes()),
            "manifest_id": manifest.task_adapter_manifest_id,
            "proof_digests": {
                proof_ref: tree_or_blob_digest(payload)
                for proof_ref, payload in proof_objects.items()
            },
            "publisher_attestation": manifest.publisher_attestation,
            "tree_hash": manifest.tree_hash,
        }
        if verification != expected:
            raise TaskAdapterStoreError("authority rejected publisher verification")

    def verify_activation(self, *, activation, authority_envelope):
        envelope = parse_json_bytes(authority_envelope)
        if envelope != {
            "scope_contract_id": activation.scope_contract_id,
            "task_adapter_id": activation.task_adapter_id,
            "task_family_id": activation.task_family_id,
            "verification_receipt_id": activation.verification_receipt_id,
        }:
            raise TaskAdapterStoreError("authority rejected activation")


class _AuthorityV2(_Authority):
    authority_version = "test.task_adapter_authority.v2"


def _store(
    tmp_path: Path,
    authority=None,
    *,
    historical_authorities=(),
) -> TaskAdapterPackageStore:
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    state_root = (tmp_path / "state").absolute()
    state_root.mkdir(mode=0o700, exist_ok=True)
    configured_authority = _Authority() if authority is None else authority
    authorities = tuple(
        sorted(
            (*historical_authorities, configured_authority),
            key=lambda item: (item.authority_id, item.authority_version),
        )
    )
    trusted_authorities = tuple(
        TaskAdapterAuthorityTrustSettings(
            authority_id=item.authority_id,
            authority_version=item.authority_version,
        )
        for item in authorities
    )
    task_adapter_settings = replace(
        settings.expert.task_adapters,
        state_path="task-adapters",
        active_authority=TaskAdapterAuthorityTrustSettings(
            authority_id=configured_authority.authority_id,
            authority_version=configured_authority.authority_version,
        ),
        trusted_authorities=trusted_authorities,
    )
    registry = TaskAdapterAuthorityRegistry(task_adapter_settings, authorities)
    return TaskAdapterPackageStore(
        (state_root / "task-adapters").absolute(),
        state_root,
        task_adapter_settings,
        registry,
    )


def _activation_envelope(package) -> bytes:
    manifest = package.manifest
    return canonical_json_bytes(
        {
            "scope_contract_id": manifest.scope_contract_id,
            "task_adapter_id": manifest.task_adapter_id,
            "task_family_id": manifest.task_family_id,
            "verification_receipt_id": (
                package.verification_receipt.verification_receipt_id
            ),
        }
    )


def test_real_archive_round_trip_and_identical_publication_replay(tmp_path):
    source = b"def evaluate(value):\n    return value + 1\n"
    manifest = _manifest(source)
    package = _package(tmp_path, manifest)
    store = _store(tmp_path)

    published = store.publish(package)
    replay = store.publish(package)
    exact = store.resolve_exact(
        task_adapter_manifest_id=manifest.task_adapter_manifest_id,
        verification_receipt_id=(
            published.verification_receipt.verification_receipt_id
        ),
    )

    assert replay == published
    assert exact == published
    assert exact.source_contents == {
        "adapter.py": source,
        "requirements.lock": RUNTIME_LOCK,
    }
    assert set(exact.dependency_ids) == {
        exact.verification_receipt.verification_receipt_id,
        exact.verification_receipt.source_extraction_receipt_id,
        exact.manifest.sanitation_report_id,
        *exact.verification_receipt.proof_object_ids,
    }
    assert len(exact.verification_receipt.proof_object_ids) == len(
        exact.verification_receipt.proof_object_digests
    )
    assert len(tuple((store.state_path / "packages").iterdir())) == 1


def test_release_matrix_artifact_bytes_are_part_of_the_verified_package(tmp_path):
    source = b"def evaluate(value):\n    return value + 1\n"
    artifact_payload = b'{"rows": [1, 2, 3]}\n'
    artifact = _starting_artifact(artifact_payload)
    extra_files = {
        "release_matrix_assets/seed/fixture.json": artifact_payload,
    }
    manifest = _manifest(
        source,
        extra_files=extra_files,
        starting_artifacts=(artifact,),
    )
    exact = _store(tmp_path).publish(
        _package(tmp_path, manifest, extra_files=extra_files)
    )

    assert exact.manifest.release_matrix_cases[0].starting_artifacts == (artifact,)
    assert (
        exact.source_contents["release_matrix_assets/seed/fixture.json"]
        == artifact_payload
    )


def test_release_matrix_artifact_closure_rejects_hidden_or_substituted_bytes(
    tmp_path,
):
    source = b"def evaluate(value):\n    return value + 1\n"
    declared_payload = b'{"rows": [1]}\n'
    artifact = _starting_artifact(declared_payload)
    hidden_files = {
        "release_matrix_assets/seed/fixture.json": declared_payload,
        "release_matrix_assets/seed/hidden.json": b"{}\n",
    }
    hidden_manifest = _manifest(
        source,
        extra_files=hidden_files,
        starting_artifacts=(artifact,),
    )
    store = _store(tmp_path)
    with pytest.raises(ContractValidationError, match="closure is not exact"):
        store.publish(_package(tmp_path, hidden_manifest, extra_files=hidden_files))

    substituted_files = {
        "release_matrix_assets/seed/fixture.json": b'{"rows": [2]}\n',
    }
    substituted_manifest = _manifest(
        source,
        publisher="publisher-b",
        extra_files=substituted_files,
        starting_artifacts=(artifact,),
    )
    with pytest.raises(ContractValidationError, match="differs from its manifest"):
        store.publish(
            _package(
                tmp_path,
                substituted_manifest,
                extra_files=substituted_files,
            )
        )


def test_release_matrix_artifact_roots_cannot_overlap(tmp_path):
    source = b"def evaluate(value):\n    return value + 1\n"
    first = _starting_artifact(
        b"first\n",
        label="first",
        package_source_root="release_matrix_assets/shared",
    )
    second = _starting_artifact(
        b"second\n",
        label="second",
        package_source_root="release_matrix_assets/shared/nested",
    )

    with pytest.raises(ContractValidationError, match="asset roots overlap"):
        _manifest(source, starting_artifacts=(first, second))
    with pytest.raises(ContractValidationError, match="workspace root"):
        replace(first, mount_path=".")


def test_runtime_files_cannot_alias_release_matrix_assets():
    source = b"def evaluate(value):\n    return value + 1\n"
    manifest = _manifest(source)

    with pytest.raises(ContractValidationError, match="asset subtree"):
        replace(
            manifest,
            task_evaluator=replace(
                manifest.task_evaluator,
                executable_path="release_matrix_assets/seed/fixture.json",
            ),
        )


def test_exact_replay_rejects_an_exhausted_materialization_budget_before_read(
    tmp_path,
    monkeypatch,
):
    source = b"def evaluate(value):\n    return value + 1\n"
    manifest = _manifest(source)
    store = _store(tmp_path)
    published = store.publish(_package(tmp_path, manifest))

    def unexpected_read(_package_path):
        raise AssertionError("bounded resolution read an over-budget package")

    monkeypatch.setattr(store, "_read_package", unexpected_read)
    with pytest.raises(TaskAdapterStoreError, match="remaining replay"):
        store.resolve_exact_bounded(
            task_adapter_manifest_id=manifest.task_adapter_manifest_id,
            verification_receipt_id=(
                published.verification_receipt.verification_receipt_id
            ),
            maximum_entries=1,
            maximum_bytes=1,
            timeout_seconds=1,
        )


def test_exact_replay_accepts_the_canonical_materialization_usage_bound(tmp_path):
    source = b"def evaluate(value):\n    return value + 1\n"
    manifest = _manifest(source)
    store = _store(tmp_path)
    published = store.publish(_package(tmp_path, manifest))
    entry_count, byte_count = task_adapter_materialization_usage(
        source_file_sizes=tuple(
            descriptor.size
            for descriptor in published.source_extraction_receipt.source_tree_files
        ),
        source_archive_sizes=(len(published.source_archive),),
        proof_object_sizes=tuple(
            len(payload) for payload in published.proof_objects.values()
        ),
        publisher_verification_sizes=(len(published.publisher_verification),),
    )

    resolved = store.resolve_exact_bounded(
        task_adapter_manifest_id=manifest.task_adapter_manifest_id,
        verification_receipt_id=(
            published.verification_receipt.verification_receipt_id
        ),
        maximum_entries=entry_count,
        maximum_bytes=byte_count,
        timeout_seconds=1,
    )

    assert resolved == published


def test_bounded_package_reads_enforce_the_materialization_deadline(
    tmp_path,
    monkeypatch,
):
    source = b"def evaluate(value):\n    return value + 1\n"
    manifest = _manifest(source)
    store = _store(tmp_path)
    published = store.publish(_package(tmp_path, manifest))
    manifest_path = (
        store._package_path(published.verification_receipt.verification_receipt_id)
        / "manifest.json"
    )
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(task_adapter_storage.time, "monotonic", lambda: next(clock))

    with pytest.raises(TaskAdapterStoreError, match="deadline expired"):
        store._read_bounded(
            manifest_path,
            deadline=1.0,
        )


def test_executable_source_mode_survives_immutable_publication(tmp_path):
    source = b"def evaluate(value):\n    return value + 1\n"
    manifest = _manifest(source, executable=True)
    store = _store(tmp_path)

    published = store.publish(_package(tmp_path, manifest, executable=True))
    source_path = (
        store._package_path(published.verification_receipt.verification_receipt_id)
        / "source"
        / "adapter.py"
    )

    assert published.source_extraction_receipt.source_tree_files[0].mode == "100755"
    assert stat.S_IMODE(source_path.stat().st_mode) == 0o555
    assert (
        store.read(published.verification_receipt.verification_receipt_id) == published
    )


def test_package_rejects_non_executable_evaluator_and_unbound_runtime_lock(tmp_path):
    source = b"def evaluate(value):\n    return value + 1\n"
    store = _store(tmp_path)
    non_executable_manifest = _manifest(source, executable=False)

    with pytest.raises(ContractValidationError, match="evaluator or runtime lock"):
        store.publish(
            _package(
                tmp_path,
                non_executable_manifest,
                executable=False,
            )
        )

    manifest = _manifest(source)
    missing_lock_manifest = _remint_manifest(
        manifest,
        runtime=replace(manifest.runtime, dependency_lock_path="missing.lock"),
    )
    with pytest.raises(ContractValidationError, match="evaluator or runtime lock"):
        store.publish(_package(tmp_path, missing_lock_manifest))

    changed_lock_manifest = _remint_manifest(
        manifest,
        runtime=replace(
            manifest.runtime,
            dependency_lock_digest=tree_or_blob_digest(b"changed-lock"),
        ),
    )
    with pytest.raises(ContractValidationError, match="runtime lock"):
        store.publish(_package(tmp_path, changed_lock_manifest))


def test_attestation_rotation_moves_active_but_preserves_exact_replay(tmp_path):
    source = b"def evaluate(value):\n    return value + 1\n"
    first_manifest = _manifest(source)
    second_manifest = replace(
        first_manifest,
        publisher_attestation={
            "publisher": "publisher-b",
            "signature": "verified-rotated",
        },
    )
    store = _store(tmp_path)
    first = store.publish(_package(tmp_path, first_manifest))
    second = store.publish(_package(tmp_path, second_manifest))

    assert (
        first.manifest.task_adapter_manifest_id
        == second.manifest.task_adapter_manifest_id
    )
    assert (
        first.verification_receipt.verification_receipt_id
        != second.verification_receipt.verification_receipt_id
    )
    first_activation = store.activate(
        scope_contract_id=first_manifest.scope_contract_id,
        task_family_id=first_manifest.task_family_id,
        task_adapter_id=first_manifest.task_adapter_id,
        verification_receipt_id=first.verification_receipt.verification_receipt_id,
        expected_activation_id=None,
        authority_envelope=_activation_envelope(first),
    )
    assert (
        store.activate(
            scope_contract_id=first_manifest.scope_contract_id,
            task_family_id=first_manifest.task_family_id,
            task_adapter_id=first_manifest.task_adapter_id,
            verification_receipt_id=first.verification_receipt.verification_receipt_id,
            expected_activation_id=None,
            authority_envelope=_activation_envelope(first),
        )
        == first_activation
    )
    second_activation = store.activate(
        scope_contract_id=second_manifest.scope_contract_id,
        task_family_id=second_manifest.task_family_id,
        task_adapter_id=second_manifest.task_adapter_id,
        verification_receipt_id=second.verification_receipt.verification_receipt_id,
        expected_activation_id=first_activation.activation_id,
        authority_envelope=_activation_envelope(second),
    )

    assert second_activation.predecessor_activation_id == first_activation.activation_id
    assert (
        store.resolve_active(
            scope_contract_id=second_manifest.scope_contract_id,
            task_family_id=second_manifest.task_family_id,
            task_adapter_id=second_manifest.task_adapter_id,
        )
        == second
    )
    assert (
        store.resolve_exact(
            task_adapter_manifest_id=first_manifest.task_adapter_manifest_id,
            verification_receipt_id=first.verification_receipt.verification_receipt_id,
        )
        == first
    )
    with pytest.raises(TaskAdapterActivationConflict):
        store.activate(
            scope_contract_id=first_manifest.scope_contract_id,
            task_family_id=first_manifest.task_family_id,
            task_adapter_id=first_manifest.task_adapter_id,
            verification_receipt_id=first.verification_receipt.verification_receipt_id,
            expected_activation_id=first_activation.activation_id,
            authority_envelope=_activation_envelope(first),
        )


def test_tampered_proof_and_wrong_authority_fail_loud(tmp_path):
    source = b"def evaluate(value):\n    return value + 1\n"
    manifest = _manifest(source)
    store = _store(tmp_path)
    published = store.publish(_package(tmp_path, manifest))
    receipt_id = published.verification_receipt.verification_receipt_id

    class _WrongAuthority(_Authority):
        authority_id = "wrong_task_adapter_authority"

    with pytest.raises(TaskAdapterStoreError, match="authority"):
        _store(tmp_path, _WrongAuthority()).read(receipt_id)

    proof_digest = next(
        iter(published.verification_receipt.proof_object_digests.values())
    )
    proof_path = store._package_path(receipt_id) / "proofs" / f"{proof_digest[7:]}.bin"
    proof_path.chmod(0o600)
    proof_path.write_bytes(json.dumps({"outcome": "passed"}).encode("utf-8"))
    with pytest.raises(TaskAdapterStoreError):
        store.read(receipt_id)


def test_abandoned_private_staging_is_removed_before_resolution(tmp_path):
    source = b"def evaluate(value):\n    return value + 1\n"
    manifest = _manifest(source)
    store = _store(tmp_path)
    published = store.publish(_package(tmp_path, manifest))
    abandoned = store.state_path / "packages" / ".staging-package-crashed"
    abandoned.mkdir(mode=0o700)
    (abandoned / "partial").write_bytes(b"partial")

    reopened = _store(tmp_path).read(
        published.verification_receipt.verification_receipt_id
    )

    assert reopened == published
    assert not abandoned.exists()


def test_cross_binding_pointer_cannot_splice_activation_lineage(tmp_path):
    source = b"def evaluate(value):\n    return value + 1\n"
    first_manifest = _manifest(source)
    second_manifest = _manifest(
        source,
        task_family_id="second_family",
        task_adapter_id="second_adapter",
    )
    store = _store(tmp_path)
    first = store.publish(_package(tmp_path, first_manifest))
    second = store.publish(_package(tmp_path, second_manifest))
    second_activation = store.activate(
        scope_contract_id=second_manifest.scope_contract_id,
        task_family_id=second_manifest.task_family_id,
        task_adapter_id=second_manifest.task_adapter_id,
        verification_receipt_id=second.verification_receipt.verification_receipt_id,
        expected_activation_id=None,
        authority_envelope=_activation_envelope(second),
    )
    first_pointer = store._active_pointer_path(
        first_manifest.scope_contract_id,
        first_manifest.task_family_id,
        first_manifest.task_adapter_id,
    )
    store._write_active_pointer(first_pointer, second_activation)

    with pytest.raises(TaskAdapterStoreError, match="another logical binding"):
        store.activate(
            scope_contract_id=first_manifest.scope_contract_id,
            task_family_id=first_manifest.task_family_id,
            task_adapter_id=first_manifest.task_adapter_id,
            verification_receipt_id=first.verification_receipt.verification_receipt_id,
            expected_activation_id=second_activation.activation_id,
            authority_envelope=_activation_envelope(first),
        )


def test_trusted_verifier_rotation_preserves_history_and_removal_revokes_it(tmp_path):
    source = b"def evaluate(value):\n    return value + 1\n"
    first_manifest = _manifest(source)
    first_store = _store(tmp_path)
    first = first_store.publish(_package(tmp_path, first_manifest))
    first_activation = first_store.activate(
        scope_contract_id=first_manifest.scope_contract_id,
        task_family_id=first_manifest.task_family_id,
        task_adapter_id=first_manifest.task_adapter_id,
        verification_receipt_id=first.verification_receipt.verification_receipt_id,
        expected_activation_id=None,
        authority_envelope=_activation_envelope(first),
    )

    rotated_store = _store(
        tmp_path,
        _AuthorityV2(),
        historical_authorities=(_Authority(),),
    )
    assert (
        rotated_store.read(first.verification_receipt.verification_receipt_id) == first
    )
    second_manifest = replace(
        first_manifest,
        publisher_attestation={
            "publisher": "publisher-b",
            "signature": "verified-rotated",
        },
    )
    second = rotated_store.publish(_package(tmp_path, second_manifest))
    second_activation = rotated_store.activate(
        scope_contract_id=second_manifest.scope_contract_id,
        task_family_id=second_manifest.task_family_id,
        task_adapter_id=second_manifest.task_adapter_id,
        verification_receipt_id=second.verification_receipt.verification_receipt_id,
        expected_activation_id=first_activation.activation_id,
        authority_envelope=_activation_envelope(second),
    )

    assert second_activation.predecessor_activation_id == first_activation.activation_id
    revoked_store = _store(tmp_path, _AuthorityV2())
    with pytest.raises(TaskAdapterStoreError, match="untrusted or revoked"):
        revoked_store.read(first.verification_receipt.verification_receipt_id)
    assert (
        revoked_store.resolve_active(
            scope_contract_id=second_manifest.scope_contract_id,
            task_family_id=second_manifest.task_family_id,
            task_adapter_id=second_manifest.task_adapter_id,
        )
        == second
    )


def test_concrete_store_replays_eligibility_pin_after_active_rotation(tmp_path):
    candidates = candidate_store(tmp_path)
    stored_candidate = candidates.persist(bootstrap_candidate_closure())
    binding = stored_candidate.closure.trigger_packet.active_task_bindings[0]
    source = b"def evaluate(value):\n    return value + 1\n"
    first_manifest = _manifest(
        source,
        scope_contract_id=stored_candidate.closure.manifest.scope_contract_id,
        task_family_id=binding.task_family_id,
        task_adapter_id=binding.task_adapter_id,
        transfer_dimensions={
            schema.dimension_id: "fixture"
            for schema in stored_candidate.closure.trigger_packet.scope_contract.context_dimension_schemas
        },
    )
    second_manifest = replace(
        first_manifest,
        publisher_attestation={
            "publisher": "publisher-b",
            "signature": "verified-rotated",
        },
    )
    store = _store(tmp_path)
    first = store.publish(_package(tmp_path, first_manifest))
    first_activation = store.activate(
        scope_contract_id=first_manifest.scope_contract_id,
        task_family_id=first_manifest.task_family_id,
        task_adapter_id=first_manifest.task_adapter_id,
        verification_receipt_id=first.verification_receipt.verification_receipt_id,
        expected_activation_id=None,
        authority_envelope=_activation_envelope(first),
    )
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    evaluator = ExpertCandidateEligibilityEvaluator(
        settings.expert.validation,
        candidates,
        store,
        _CurrentReleaseProvider(None),
    )
    eligibility = evaluator.decide(
        candidate_id=stored_candidate.closure.manifest.candidate_id
    )

    second = store.publish(_package(tmp_path, second_manifest))
    store.activate(
        scope_contract_id=second_manifest.scope_contract_id,
        task_family_id=second_manifest.task_family_id,
        task_adapter_id=second_manifest.task_adapter_id,
        verification_receipt_id=second.verification_receipt.verification_receipt_id,
        expected_activation_id=first_activation.activation_id,
        authority_envelope=_activation_envelope(second),
    )

    replayed = evaluator.replay(
        candidate_id=stored_candidate.closure.manifest.candidate_id,
        task_adapter_pins=eligibility.decision.task_adapter_pins,
    )
    current = evaluator.decide(
        candidate_id=stored_candidate.closure.manifest.candidate_id
    )
    assert replayed == eligibility
    assert (
        current.decision.task_adapter_pins[0].verification_receipt_id
        == second.verification_receipt.verification_receipt_id
    )
    assert current != eligibility
