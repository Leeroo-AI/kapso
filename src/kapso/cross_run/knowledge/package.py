"""Deterministic, proof-closed knowledge snapshot package assembly."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import tempfile
from collections.abc import Callable, Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    freeze_json,
    parse_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.catalog.store import CatalogGenerationManifest
from kapso.cross_run.contracts import (
    AdmissionState,
    CatalogEntryState,
    EmbeddingSidecar,
    ExpertScopeContract,
    KnowledgeSnapshotManifest,
    MissingReferenceError,
    StrictContract,
)
from kapso.cross_run.knowledge.index import SnapshotSearchIndex
from kapso.cross_run.record_registry import (
    parse_knowledge_record_payload,
    record_identity,
)

_SNAPSHOT_MANIFEST_PATH = "snapshot.json"
_SNAPSHOT_KIND_PATH = "snapshot-kind.json"
_SCOPE_CONTRACT_PATH = "scope-contract.json"
_CATALOG_GENERATION_PATH = "catalog-generation.json"
_INDEX_DIRECTORY = "index"
_EMPTY_KIND = "EMPTY"
_CATALOG_KIND = "CATALOG"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class KnowledgeSnapshotPackageError(ValueError):
    """A snapshot package is incomplete, corrupt, or unsafe to materialize."""


def _record_id(record: StrictContract) -> str:
    identity_field = record.IDENTITY_FIELD
    if identity_field is None:
        raise KnowledgeSnapshotPackageError("snapshot record has no content identity")
    return require_content_id(getattr(record, identity_field), identity_field)


def _record_kind(record_id: str) -> str:
    return record_id.split(":sha256:", 1)[0]


def _record_path(record_id: str) -> str:
    namespace, digest = record_id.split(":sha256:", 1)
    return f"records/{namespace}/{digest}.json"


def _record_id_from_path(relative_path: str) -> str:
    parts = PurePosixPath(relative_path).parts
    if len(parts) != 3 or parts[0] != "records" or not parts[2].endswith(".json"):
        raise KnowledgeSnapshotPackageError("snapshot record path is invalid")
    record_id = f"{parts[1]}:sha256:{parts[2].removesuffix('.json')}"
    require_content_id(record_id, "snapshot record path identity")
    if _record_path(record_id) != relative_path:
        raise KnowledgeSnapshotPackageError("snapshot record path is not canonical")
    return record_id


def _validate_relative_path(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise KnowledgeSnapshotPackageError(f"{name} must be a relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise KnowledgeSnapshotPackageError(f"{name} is not a normalized path")


def _freeze_files(files: Mapping[str, bytes], name: str) -> Mapping[str, bytes]:
    if not isinstance(files, MappingABC):
        raise KnowledgeSnapshotPackageError(f"{name} must be a mapping")
    frozen: dict[str, bytes] = {}
    for relative_path in sorted(files):
        _validate_relative_path(relative_path, f"{name} key")
        payload = files[relative_path]
        if not isinstance(payload, bytes):
            raise KnowledgeSnapshotPackageError(f"{name} values must be bytes")
        frozen[relative_path] = payload
    return MappingProxyType(frozen)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    os.fsync(descriptor)
    os.close(descriptor)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one directory only when the destination is absent."""

    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise KnowledgeSnapshotPackageError(
            "atomic no-replace directory publication is unavailable"
        )
    rename_at2 = libc.renameat2
    rename_at2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename_at2.restype = ctypes.c_int
    result = rename_at2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise KnowledgeSnapshotPackageError("materialization target already exists")
    raise OSError(error_number, os.strerror(error_number), destination)


def _record_envelope(record: StrictContract) -> Mapping[str, Any]:
    record_id = _record_id(record)
    return freeze_json(
        {
            "record_id": record_id,
            "record_kind": _record_kind(record_id),
            "payload": record.to_dict(),
        },
        "knowledge record envelope",
    )


def _record_envelope_payload(
    record_kind: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    record_id = record_identity(parse_knowledge_record_payload(record_kind, payload))
    return freeze_json(
        {
            "record_id": record_id,
            "record_kind": record_kind,
            "payload": payload,
        },
        "knowledge record envelope",
    )


def _validate_canonical_record_bytes(record: StrictContract, payload: bytes) -> None:
    if payload != record.to_json_bytes():
        raise KnowledgeSnapshotPackageError(
            f"record bytes are not canonical for {_record_id(record)}"
        )


@dataclass(frozen=True)
class PreparedKnowledgeSnapshot:
    """Canonical catalog closure prepared before search sidecars are built."""

    scope_contract: ExpertScopeContract
    catalog_generation: CatalogGenerationManifest
    snapshot_kind: str
    files: Mapping[str, bytes]
    record_envelopes: tuple[Mapping[str, Any], ...]
    retrieval_root_ids: tuple[str, ...]
    admitted_episode_ids: tuple[str, ...]
    admitted_prior_idea_ids: tuple[str, ...]
    active_claim_revision_ids: tuple[str, ...]
    entry_state_ids: tuple[str, ...]
    included_assertion_ids: tuple[str, ...]
    included_revocation_ids: tuple[str, ...]
    included_bundle_ids: tuple[str, ...]
    record_closure_digest: str
    proof_dependencies: Mapping[str, tuple[str, ...]]
    _records_by_id: Mapping[str, Mapping[str, Any]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", _freeze_files(self.files, "prepared files"))
        if not isinstance(self.record_envelopes, tuple):
            raise KnowledgeSnapshotPackageError("record_envelopes must be a tuple")
        frozen_envelopes: list[Mapping[str, Any]] = []
        for envelope in self.record_envelopes:
            if not isinstance(envelope, MappingABC) or set(envelope) != {
                "payload",
                "record_id",
                "record_kind",
            }:
                raise KnowledgeSnapshotPackageError(
                    "record envelope fields are invalid"
                )
            record_id = record_identity(
                parse_knowledge_record_payload(
                    envelope["record_kind"],
                    envelope["payload"],
                )
            )
            if record_id != envelope["record_id"]:
                raise KnowledgeSnapshotPackageError(
                    "record envelope identity differs from its payload"
                )
            frozen_envelopes.append(freeze_json(envelope, "prepared record envelope"))
        object.__setattr__(self, "record_envelopes", tuple(frozen_envelopes))
        if self.snapshot_kind not in {_EMPTY_KIND, _CATALOG_KIND}:
            raise KnowledgeSnapshotPackageError("snapshot kind is invalid")
        for name in (
            "retrieval_root_ids",
            "admitted_episode_ids",
            "admitted_prior_idea_ids",
            "active_claim_revision_ids",
            "entry_state_ids",
            "included_assertion_ids",
            "included_revocation_ids",
            "included_bundle_ids",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise KnowledgeSnapshotPackageError(f"{name} must be sorted and unique")
            for value in values:
                require_content_id(value, name)
        envelope_ids = tuple(
            envelope["record_id"] for envelope in self.record_envelopes
        )
        if envelope_ids != tuple(sorted(set(envelope_ids))):
            raise KnowledgeSnapshotPackageError(
                "prepared record envelopes must be sorted and unique"
            )
        object.__setattr__(
            self,
            "_records_by_id",
            MappingProxyType(
                {envelope["record_id"]: envelope for envelope in self.record_envelopes}
            ),
        )
        expected_digest = tree_or_blob_digest(
            canonical_json_bytes(self.record_envelopes)
        )
        if self.record_closure_digest != expected_digest:
            raise KnowledgeSnapshotPackageError("record closure digest differs")
        if not set(self.retrieval_root_ids).issubset(envelope_ids):
            raise MissingReferenceError("retrieval root leaves the record closure")
        expected_dependencies = _build_query_proof_dependencies(
            self.record_envelopes,
            self.catalog_generation,
        )
        if canonical_json_bytes(self.proof_dependencies) != canonical_json_bytes(
            expected_dependencies
        ):
            raise KnowledgeSnapshotPackageError(
                "prepared query proof dependencies differ from canonical records"
            )
        expected_roots = tuple(
            sorted(
                (
                    *self.admitted_episode_ids,
                    *self.admitted_prior_idea_ids,
                    *self.active_claim_revision_ids,
                )
            )
        )
        if self.retrieval_root_ids != expected_roots:
            raise KnowledgeSnapshotPackageError(
                "retrieval roots differ from admitted active records"
            )
        declared_ids = {
            *self.entry_state_ids,
            *self.included_assertion_ids,
            *self.included_revocation_ids,
            *self.included_bundle_ids,
        }
        if not declared_ids.issubset(envelope_ids):
            raise MissingReferenceError(
                "prepared manifest field leaves the record closure"
            )
        expected_kind = (
            _EMPTY_KIND
            if self.catalog_generation.generation_number == 0
            else _CATALOG_KIND
        )
        if self.snapshot_kind != expected_kind:
            raise KnowledgeSnapshotPackageError(
                "snapshot kind differs from catalog generation"
            )

    @property
    def catalog_generation_id(self) -> str:
        return self.catalog_generation.catalog_generation_id

    @property
    def scope_contract_id(self) -> str:
        return self.scope_contract.scope_contract_id

    def record_by_id(self, record_id: str) -> Mapping[str, Any]:
        require_content_id(record_id, "record_id")
        record = self._records_by_id.get(record_id)
        if record is None:
            raise MissingReferenceError("record is absent from the prepared closure")
        return record


@dataclass(frozen=True)
class KnowledgeSnapshotPackage:
    """One verified manifest and every byte it declares."""

    manifest: KnowledgeSnapshotManifest
    prepared: PreparedKnowledgeSnapshot
    files: Mapping[str, bytes]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", _freeze_files(self.files, "package files"))
        self.verify()

    @property
    def record_envelopes(self) -> tuple[Mapping[str, Any], ...]:
        return self.prepared.record_envelopes

    @property
    def record_closure_digest(self) -> str:
        return self.prepared.record_closure_digest

    @property
    def retrieval_root_ids(self) -> tuple[str, ...]:
        return self.prepared.retrieval_root_ids

    def record_by_id(self, record_id: str) -> Mapping[str, Any]:
        return self.prepared.record_by_id(record_id)

    def verify(self) -> None:
        manifest, prepared = _verify_package_files(self.files)
        if manifest != self.manifest or prepared != self.prepared:
            raise KnowledgeSnapshotPackageError(
                "package objects differ from their canonical file closure"
            )

    def materialize(self, target: Path | str) -> Path:
        """Atomically create a new local package directory from verified bytes."""

        self.verify()
        destination = Path(target)
        if not destination.is_absolute() or destination != destination.absolute():
            raise KnowledgeSnapshotPackageError(
                "materialization target must be an absolute normalized path"
            )
        if destination.exists() or destination.is_symlink():
            raise KnowledgeSnapshotPackageError("materialization target already exists")
        parent = destination.parent
        if (
            parent.is_symlink()
            or not parent.is_dir()
            or parent.resolve() != parent.absolute()
        ):
            raise KnowledgeSnapshotPackageError(
                "materialization parent must be a real directory"
            )
        with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=parent) as staging:
            staging_root = Path(staging)
            for relative_path, payload in self.files.items():
                output = staging_root / relative_path
                output.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    output,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o444,
                )
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            reopened = KnowledgeSnapshotPackage.open(staging_root)
            if reopened.manifest.snapshot_id != self.manifest.snapshot_id:
                raise KnowledgeSnapshotPackageError(
                    "materialized package identity changed before commit"
                )
            directories = tuple(
                sorted(
                    (path for path in staging_root.rglob("*") if path.is_dir()),
                    key=lambda path: len(path.parts),
                    reverse=True,
                )
            )
            for directory in directories:
                _fsync_directory(directory)
            _fsync_directory(staging_root)
            _rename_directory_no_replace(staging_root, destination)
            _fsync_directory(parent)
        return destination

    @classmethod
    def open(cls, root: Path | str) -> KnowledgeSnapshotPackage:
        """Read one clean materialized package and verify every declared byte."""

        package_root = Path(root)
        if (
            not package_root.is_absolute()
            or package_root != package_root.absolute()
            or package_root.is_symlink()
            or not package_root.is_dir()
            or package_root.resolve() != package_root.absolute()
        ):
            raise KnowledgeSnapshotPackageError(
                "package root must be an absolute real directory"
            )
        files: dict[str, bytes] = {}
        for path in sorted(package_root.rglob("*")):
            relative_path = path.relative_to(package_root).as_posix()
            if path.is_symlink():
                raise KnowledgeSnapshotPackageError("package cannot contain symlinks")
            metadata = path.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise KnowledgeSnapshotPackageError(
                    "package entries must be regular files"
                )
            files[relative_path] = path.read_bytes()
        manifest, prepared = _verify_package_files(files)
        return cls(manifest=manifest, prepared=prepared, files=files)


class KnowledgeSnapshotPackageBuilder:
    """Prepare catalog truth first, then bind rebuildable sidecars at finalize."""

    @staticmethod
    def prepare(
        scope_contract: ExpertScopeContract,
        catalog_generation: CatalogGenerationManifest,
        read_object_bytes: Callable[[str], bytes],
    ) -> PreparedKnowledgeSnapshot:
        if not isinstance(catalog_generation, CatalogGenerationManifest):
            raise TypeError("catalog_generation must be CatalogGenerationManifest")
        generation = catalog_generation
        if generation.scope_contract_id != scope_contract.scope_contract_id:
            raise KnowledgeSnapshotPackageError(
                "catalog generation uses another scope contract"
            )
        envelopes_by_id: dict[str, Mapping[str, Any]] = {
            scope_contract.scope_contract_id: _record_envelope(scope_contract),
            generation.catalog_generation_id: _record_envelope(generation),
        }
        bytes_by_id = {
            scope_contract.scope_contract_id: scope_contract.to_json_bytes(),
            generation.catalog_generation_id: generation.to_json_bytes(),
        }
        for object_id in (
            *generation.fact_object_ids,
            *generation.applied_input_delta_ids,
        ):
            payload_bytes = read_object_bytes(object_id)
            payload = parse_json_bytes(payload_bytes)
            if not isinstance(payload, MappingABC):
                raise KnowledgeSnapshotPackageError("catalog fact must be an object")
            if payload_bytes != canonical_json_bytes(payload):
                raise KnowledgeSnapshotPackageError(
                    f"catalog fact bytes are not canonical: {object_id}"
                )
            record_kind = _record_kind(object_id)
            envelope = _record_envelope_payload(record_kind, payload)
            if envelope["record_id"] != object_id:
                raise KnowledgeSnapshotPackageError(
                    "catalog fact bytes do not own their generation ID"
                )
            envelopes_by_id[object_id] = envelope
            bytes_by_id[object_id] = payload_bytes
        current_states = tuple(
            CatalogEntryState.from_json_bytes(read_object_bytes(state_id))
            for state_id in generation.derived_object_ids
        )
        states_by_id = {state.catalog_entry_state_id: state for state in current_states}
        pending_predecessor_ids = sorted(
            state.predecessor_state_id
            for state in current_states
            if state.predecessor_state_id is not None
        )
        while pending_predecessor_ids:
            state_id = pending_predecessor_ids.pop(0)
            if state_id in states_by_id:
                continue
            state = CatalogEntryState.from_json_bytes(read_object_bytes(state_id))
            if state.catalog_entry_state_id != state_id:
                raise KnowledgeSnapshotPackageError(
                    "predecessor state bytes do not own their referenced ID"
                )
            states_by_id[state_id] = state
            if state.predecessor_state_id is not None:
                pending_predecessor_ids.append(state.predecessor_state_id)
                pending_predecessor_ids.sort()
        states = tuple(states_by_id[state_id] for state_id in sorted(states_by_id))
        for state in states:
            state_id = state.catalog_entry_state_id
            state_bytes = read_object_bytes(state_id)
            _validate_canonical_record_bytes(state, state_bytes)
            envelopes_by_id[state_id] = _record_envelope(state)
            bytes_by_id[state_id] = state_bytes
        return _prepare_snapshot(
            scope_contract,
            generation,
            envelopes_by_id,
            bytes_by_id,
            states,
        )

    @staticmethod
    def prepare_empty(
        scope_contract: ExpertScopeContract,
        catalog_generation: CatalogGenerationManifest,
    ) -> PreparedKnowledgeSnapshot:
        if catalog_generation.generation_number != 0:
            raise KnowledgeSnapshotPackageError(
                "an explicit EMPTY snapshot requires catalog generation zero"
            )
        return KnowledgeSnapshotPackageBuilder.prepare(
            scope_contract,
            catalog_generation,
            lambda object_id: _missing_empty_object(object_id),
        )

    @staticmethod
    def finalize(
        prepared: PreparedKnowledgeSnapshot,
        *,
        parent_snapshot_ids: tuple[str, ...],
        sanitation_policy_version: str,
        retrieval_policy_version: str,
        configuration_fingerprint: str,
        prompt_budget_policy: Mapping[str, Any],
        published_at: str,
        publisher_attestation: Mapping[str, Any],
        search_files: Mapping[str, bytes] = MappingProxyType({}),
        embedding_sidecars: tuple[EmbeddingSidecar, ...] = (),
    ) -> KnowledgeSnapshotPackage:
        frozen_search = _freeze_files(search_files, "search files")
        for relative_path in frozen_search:
            if PurePosixPath(relative_path).parts[0] != _INDEX_DIRECTORY:
                raise KnowledgeSnapshotPackageError(
                    "search sidecars must live under index/"
                )
            if relative_path in prepared.files:
                raise KnowledgeSnapshotPackageError(
                    "search sidecar collides with canonical package content"
                )
        package_content = {**prepared.files, **frozen_search}
        checksums = {
            path: tree_or_blob_digest(payload)
            for path, payload in sorted(package_content.items())
        }
        manifest = KnowledgeSnapshotManifest.mint(
            scope_contract_id=prepared.scope_contract.scope_contract_id,
            scope_id=prepared.scope_contract.scope_id,
            parent_snapshot_ids=tuple(sorted(parent_snapshot_ids)),
            included_bundle_ids=prepared.included_bundle_ids,
            admitted_episode_ids=prepared.admitted_episode_ids,
            admitted_prior_idea_ids=prepared.admitted_prior_idea_ids,
            active_claim_revision_ids=prepared.active_claim_revision_ids,
            catalog_generation=prepared.catalog_generation.generation_number,
            configuration_fingerprint=configuration_fingerprint,
            entry_state_refs=prepared.entry_state_ids,
            included_assertion_ids=prepared.included_assertion_ids,
            included_revocation_ids=prepared.included_revocation_ids,
            proof_dependency_closure_ids=tuple(
                record["record_id"] for record in prepared.record_envelopes
            ),
            sanitation_policy_version=sanitation_policy_version,
            retrieval_policy_version=retrieval_policy_version,
            embedding_sidecars=tuple(
                sorted(
                    embedding_sidecars,
                    key=lambda sidecar: sidecar.embedding_space_id,
                )
            ),
            prompt_budget_policy=prompt_budget_policy,
            checksums=checksums,
            published_at=published_at,
            publisher_attestation=publisher_attestation,
        )
        files = {
            _SNAPSHOT_MANIFEST_PATH: manifest.to_json_bytes(),
            **package_content,
        }
        return KnowledgeSnapshotPackage(
            manifest=manifest,
            prepared=prepared,
            files=files,
        )

    @staticmethod
    def build(
        scope_contract: ExpertScopeContract,
        catalog_generation: CatalogGenerationManifest,
        read_object_bytes: Callable[[str], bytes],
        **finalize_fields: Any,
    ) -> KnowledgeSnapshotPackage:
        prepared = KnowledgeSnapshotPackageBuilder.prepare(
            scope_contract,
            catalog_generation,
            read_object_bytes,
        )
        return KnowledgeSnapshotPackageBuilder.finalize(
            prepared,
            **finalize_fields,
        )


def _missing_empty_object(object_id: str) -> bytes:
    raise MissingReferenceError(
        f"EMPTY generation unexpectedly requested object {object_id}"
    )


def _optional_reference(payload: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    value = payload[field_name]
    if value is None:
        return ()
    return (require_content_id(value, field_name),)


def _reference_tuple(payload: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    values = tuple(payload[field_name])
    for value in values:
        require_content_id(value, field_name)
    return values


def _typed_required_references(envelope: Mapping[str, Any]) -> tuple[str, ...]:
    kind = envelope["record_kind"]
    payload = envelope["payload"]
    references: tuple[str, ...]
    if kind == "transfer-episode":
        references = (
            payload["source_bundle_id"],
            payload["sanitation_report_id"],
            *_optional_reference(payload, "supersedes_projection_id"),
            *_optional_reference(payload, "parent_episode_ref"),
            *_reference_tuple(payload, "derivation_refs"),
        )
    elif kind == "prior-idea":
        references = (
            payload["source_bundle_id"],
            payload["sanitation_report_id"],
            *_optional_reference(payload, "supersedes_projection_id"),
        )
    elif kind == "knowledge-claim-revision":
        provenance = payload["proposal_provenance"]
        operation_receipt_id = provenance.get("operation_receipt_id")
        operation_references = (
            ()
            if operation_receipt_id is None
            else (
                require_content_id(
                    operation_receipt_id,
                    "proposal_provenance.operation_receipt_id",
                ),
            )
        )
        references = (
            payload["scope_contract_id"],
            *_reference_tuple(payload, "supporting_episode_ids"),
            *_reference_tuple(payload, "contradicting_episode_ids"),
            *_reference_tuple(payload, "supersedes_revision_ids"),
            *operation_references,
        )
    elif kind == "catalog-entry-state":
        references = (
            payload["subject_payload_id"],
            *_optional_reference(payload, "predecessor_state_id"),
            *_reference_tuple(payload, "superseded_by_payload_ids"),
            *_reference_tuple(payload, "assertion_ids"),
            *_reference_tuple(payload, "revocation_ids"),
            *_reference_tuple(payload, "taint_source_ids"),
        )
    elif kind == "review-assertion":
        references = (
            payload["subject_id"],
            payload["review_operation_ref"],
            *_optional_reference(payload, "supersedes_assertion_id"),
            *_reference_tuple(payload, "exact_evidence_refs"),
        )
    elif kind == "catalog-revocation":
        references = (
            payload["subject_id"],
            *_reference_tuple(payload, "exact_evidence_refs"),
        )
    elif kind == "catalog-taint":
        references = (
            payload["subject_id"],
            payload["source_subject_id"],
            *_reference_tuple(payload, "exact_evidence_refs"),
        )
    elif kind == "claim-evidence-closure":
        references = (
            payload["claim_revision_id"],
            payload["proposer_operation_receipt_id"],
            *_reference_tuple(payload, "evaluated_episode_ids"),
        )
    elif kind == "catalog-agent-operation":
        references = (
            payload["operation_receipt_id"],
            *_reference_tuple(payload, "produced_object_ids"),
        )
    elif kind == "bundle-projection-manifest":
        references = (
            payload["source_bundle_id"],
            payload["sanitation_report_id"],
            *_reference_tuple(payload, "episode_ids"),
            *_reference_tuple(payload, "prior_idea_ids"),
            *_reference_tuple(payload, "derivation_object_ids"),
        )
    elif kind == "catalog-input-delta":
        references = (
            payload["scope_contract_id"],
            *_reference_tuple(payload, "added_object_ids"),
            *_reference_tuple(payload, "dependency_closure_ids"),
        )
    elif kind == "run-bundle":
        references = (
            payload["scope_contract_id"],
            *_optional_reference(payload, "supersedes_bundle_id"),
        )
    elif kind in {
        "catalog-generation",
        "coding-agent-operation-receipt",
        "execution-revision-event",
        "expert-scope-contract",
        "sanitation-report",
    }:
        references = ()
    else:
        raise KnowledgeSnapshotPackageError(
            f"unsupported knowledge proof record kind: {kind}"
        )
    for reference in references:
        require_content_id(reference, f"{kind} proof reference")
    return tuple(sorted(set(references)))


def _query_direct_references(envelope: Mapping[str, Any]) -> tuple[str, ...]:
    kind = envelope["record_kind"]
    payload = envelope["payload"]
    required = _typed_required_references(envelope)
    if kind == "bundle-projection-manifest":
        return tuple(
            sorted(
                {
                    payload["source_bundle_id"],
                    payload["sanitation_report_id"],
                    *payload["derivation_object_ids"],
                }
            )
        )
    if kind == "catalog-agent-operation":
        return (payload["operation_receipt_id"],)
    if kind in {"catalog-generation", "catalog-input-delta", "run-bundle"}:
        return ()
    return required


def _build_query_proof_dependencies(
    record_envelopes: tuple[Mapping[str, Any], ...],
    catalog_generation: CatalogGenerationManifest,
) -> Mapping[str, tuple[str, ...]]:
    records_by_id = {envelope["record_id"]: envelope for envelope in record_envelopes}
    dependencies = {
        record_id: set(_query_direct_references(envelope))
        for record_id, envelope in records_by_id.items()
    }
    for record_id, envelope in records_by_id.items():
        required = _typed_required_references(envelope)
        missing = set(required) - set(records_by_id)
        if missing:
            raise MissingReferenceError(
                f"typed proof dependency is absent for {record_id}: "
                f"{tuple(sorted(missing))}"
            )
    for subject_id, state_id in catalog_generation.active_entry_state_ids.items():
        dependencies[subject_id].add(state_id)
    for record_id, envelope in records_by_id.items():
        payload = envelope["payload"]
        if envelope["record_kind"] == "bundle-projection-manifest":
            for produced_id in (
                *payload["episode_ids"],
                *payload["prior_idea_ids"],
            ):
                dependencies[produced_id].add(record_id)
        elif envelope["record_kind"] == "claim-evidence-closure":
            dependencies[payload["claim_revision_id"]].add(record_id)
        elif envelope["record_kind"] == "catalog-agent-operation":
            for produced_id in payload["produced_object_ids"]:
                dependencies[produced_id].add(record_id)
    return MappingProxyType(
        {
            record_id: tuple(sorted(reference_ids))
            for record_id, reference_ids in sorted(dependencies.items())
        }
    )


def _prepare_snapshot(
    scope_contract: ExpertScopeContract,
    catalog_generation: CatalogGenerationManifest,
    envelopes_by_id: Mapping[str, Mapping[str, Any]],
    bytes_by_id: Mapping[str, bytes],
    states: tuple[CatalogEntryState, ...],
) -> PreparedKnowledgeSnapshot:
    expected_ids = {
        scope_contract.scope_contract_id,
        catalog_generation.catalog_generation_id,
        *catalog_generation.fact_object_ids,
        *catalog_generation.derived_object_ids,
        *catalog_generation.applied_input_delta_ids,
        *(state.catalog_entry_state_id for state in states),
    }
    if set(envelopes_by_id) != expected_ids or set(bytes_by_id) != expected_ids:
        raise MissingReferenceError("prepared snapshot record closure is incomplete")
    states_by_id = {state.catalog_entry_state_id: state for state in states}
    if not set(catalog_generation.derived_object_ids).issubset(states_by_id):
        raise KnowledgeSnapshotPackageError(
            "catalog derived closure contains a non-entry-state record"
        )
    active_state_ids = tuple(sorted(catalog_generation.active_entry_state_ids.values()))
    if active_state_ids != tuple(
        sorted(
            state_id
            for state_id in states_by_id
            if state_id in catalog_generation.derived_object_ids
        )
    ):
        raise MissingReferenceError("catalog active entry-state closure is incomplete")
    for state_id in catalog_generation.derived_object_ids:
        state = states_by_id[state_id]
        if (
            state.catalog_generation != catalog_generation.generation_number
            or state.configuration_fingerprint
            != catalog_generation.configuration_fingerprint
            or state.subject_payload_id not in catalog_generation.active_entry_state_ids
            or catalog_generation.active_entry_state_ids[state.subject_payload_id]
            != state.catalog_entry_state_id
        ):
            raise KnowledgeSnapshotPackageError(
                "catalog entry state does not belong to the exact generation"
            )
    for state in states:
        predecessor_id = state.predecessor_state_id
        if predecessor_id is None:
            continue
        predecessor = states_by_id.get(predecessor_id)
        if predecessor is None:
            raise MissingReferenceError("catalog predecessor state is absent")
        if (
            predecessor.subject_payload_id != state.subject_payload_id
            or predecessor.catalog_generation >= state.catalog_generation
        ):
            raise KnowledgeSnapshotPackageError(
                "catalog predecessor state lineage is invalid"
            )
    admitted_subject_ids = tuple(
        sorted(
            states_by_id[state_id].subject_payload_id
            for state_id in catalog_generation.derived_object_ids
            if states_by_id[state_id].admission_state is AdmissionState.ADMITTED
        )
    )
    admitted_episode_ids = tuple(
        record_id
        for record_id in admitted_subject_ids
        if _record_kind(record_id) == "transfer-episode"
    )
    admitted_prior_idea_ids = tuple(
        record_id
        for record_id in admitted_subject_ids
        if _record_kind(record_id) == "prior-idea"
    )
    active_claim_revision_ids = tuple(
        record_id
        for record_id in admitted_subject_ids
        if _record_kind(record_id) == "knowledge-claim-revision"
    )
    retrieval_root_ids = tuple(
        sorted(
            (
                *admitted_episode_ids,
                *admitted_prior_idea_ids,
                *active_claim_revision_ids,
            )
        )
    )
    envelopes = tuple(
        envelopes_by_id[record_id] for record_id in sorted(envelopes_by_id)
    )
    proof_dependencies = _build_query_proof_dependencies(
        envelopes,
        catalog_generation,
    )
    files: dict[str, bytes] = {
        _SNAPSHOT_KIND_PATH: canonical_json_bytes(
            {
                "snapshot_kind": (
                    _EMPTY_KIND
                    if catalog_generation.generation_number == 0
                    else _CATALOG_KIND
                )
            }
        ),
        _SCOPE_CONTRACT_PATH: scope_contract.to_json_bytes(),
        _CATALOG_GENERATION_PATH: catalog_generation.to_json_bytes(),
    }
    for record_id in sorted(envelopes_by_id):
        if record_id in {
            scope_contract.scope_contract_id,
            catalog_generation.catalog_generation_id,
        }:
            continue
        files[_record_path(record_id)] = bytes_by_id[record_id]
    snapshot_kind = (
        _EMPTY_KIND if catalog_generation.generation_number == 0 else _CATALOG_KIND
    )
    if snapshot_kind == _EMPTY_KIND and (
        catalog_generation.fact_object_ids
        or catalog_generation.derived_object_ids
        or retrieval_root_ids
    ):
        raise KnowledgeSnapshotPackageError("EMPTY snapshot contains catalog records")
    return PreparedKnowledgeSnapshot(
        scope_contract=scope_contract,
        catalog_generation=catalog_generation,
        snapshot_kind=snapshot_kind,
        files=files,
        record_envelopes=envelopes,
        retrieval_root_ids=retrieval_root_ids,
        admitted_episode_ids=admitted_episode_ids,
        admitted_prior_idea_ids=admitted_prior_idea_ids,
        active_claim_revision_ids=active_claim_revision_ids,
        entry_state_ids=active_state_ids,
        included_assertion_ids=tuple(
            record_id
            for record_id in sorted(catalog_generation.fact_object_ids)
            if _record_kind(record_id) == "review-assertion"
        ),
        included_revocation_ids=tuple(
            record_id
            for record_id in sorted(catalog_generation.fact_object_ids)
            if _record_kind(record_id) == "catalog-revocation"
        ),
        included_bundle_ids=tuple(
            record_id
            for record_id in sorted(catalog_generation.fact_object_ids)
            if _record_kind(record_id) == "run-bundle"
        ),
        record_closure_digest=tree_or_blob_digest(canonical_json_bytes(envelopes)),
        proof_dependencies=proof_dependencies,
    )


def _verify_package_files(
    files: Mapping[str, bytes],
) -> tuple[KnowledgeSnapshotManifest, PreparedKnowledgeSnapshot]:
    frozen_files = _freeze_files(files, "package files")
    manifest_bytes = frozen_files.get(_SNAPSHOT_MANIFEST_PATH)
    if manifest_bytes is None:
        raise MissingReferenceError("snapshot manifest is absent")
    manifest = KnowledgeSnapshotManifest.from_json_bytes(manifest_bytes)
    if manifest_bytes != manifest.to_json_bytes():
        raise KnowledgeSnapshotPackageError("snapshot manifest bytes are not canonical")
    declared_paths = set(manifest.checksums)
    actual_paths = set(frozen_files) - {_SNAPSHOT_MANIFEST_PATH}
    if declared_paths != actual_paths:
        raise MissingReferenceError("snapshot checksum file closure is incomplete")
    for relative_path in sorted(declared_paths):
        if (
            tree_or_blob_digest(frozen_files[relative_path])
            != manifest.checksums[relative_path]
        ):
            raise KnowledgeSnapshotPackageError(
                f"snapshot checksum mismatch: {relative_path}"
            )
    kind_payload = parse_json_bytes(frozen_files[_SNAPSHOT_KIND_PATH])
    if not isinstance(kind_payload, MappingABC) or set(kind_payload) != {
        "snapshot_kind"
    }:
        raise KnowledgeSnapshotPackageError("snapshot kind record is invalid")
    if frozen_files[_SNAPSHOT_KIND_PATH] != canonical_json_bytes(kind_payload):
        raise KnowledgeSnapshotPackageError("snapshot kind bytes are not canonical")
    scope_contract = ExpertScopeContract.from_json_bytes(
        frozen_files[_SCOPE_CONTRACT_PATH]
    )
    catalog_generation = CatalogGenerationManifest.from_json_bytes(
        frozen_files[_CATALOG_GENERATION_PATH]
    )
    if (
        scope_contract.to_json_bytes() != frozen_files[_SCOPE_CONTRACT_PATH]
        or catalog_generation.to_json_bytes() != frozen_files[_CATALOG_GENERATION_PATH]
    ):
        raise KnowledgeSnapshotPackageError(
            "scope or catalog generation bytes are not canonical"
        )
    actual_record_paths = {
        path for path in frozen_files if PurePosixPath(path).parts[0] == "records"
    }
    record_payloads = {
        _record_id_from_path(path): frozen_files[path]
        for path in sorted(actual_record_paths)
    }
    required_generation_ids = {
        *catalog_generation.fact_object_ids,
        *catalog_generation.derived_object_ids,
        *catalog_generation.applied_input_delta_ids,
    }
    if not required_generation_ids.issubset(record_payloads):
        raise MissingReferenceError("snapshot catalog record closure is incomplete")
    prepared = KnowledgeSnapshotPackageBuilder.prepare(
        scope_contract,
        catalog_generation,
        record_payloads.__getitem__,
    )
    expected_record_paths = {
        _record_path(record["record_id"])
        for record in prepared.record_envelopes
        if record["record_id"]
        not in {
            prepared.scope_contract_id,
            prepared.catalog_generation_id,
        }
    }
    if actual_record_paths != expected_record_paths:
        raise KnowledgeSnapshotPackageError(
            "snapshot contains records outside the exact catalog proof closure"
        )
    if kind_payload["snapshot_kind"] != prepared.snapshot_kind:
        raise KnowledgeSnapshotPackageError("snapshot kind differs from catalog state")
    expected_proof_ids = tuple(
        record["record_id"] for record in prepared.record_envelopes
    )
    if (
        manifest.scope_contract_id != scope_contract.scope_contract_id
        or manifest.scope_id != scope_contract.scope_id
        or manifest.catalog_generation != catalog_generation.generation_number
        or manifest.included_bundle_ids != prepared.included_bundle_ids
        or manifest.admitted_episode_ids != prepared.admitted_episode_ids
        or manifest.admitted_prior_idea_ids != prepared.admitted_prior_idea_ids
        or manifest.active_claim_revision_ids != prepared.active_claim_revision_ids
        or manifest.entry_state_refs != prepared.entry_state_ids
        or manifest.included_assertion_ids != prepared.included_assertion_ids
        or manifest.included_revocation_ids != prepared.included_revocation_ids
        or manifest.proof_dependency_closure_ids != expected_proof_ids
    ):
        raise KnowledgeSnapshotPackageError(
            "snapshot manifest differs from its exact catalog closure"
        )
    index_files = {
        path: payload
        for path, payload in frozen_files.items()
        if PurePosixPath(path).parts[0] == _INDEX_DIRECTORY
    }
    if index_files:
        search_index = SnapshotSearchIndex.open(prepared, index_files)
        search_index.verify(manifest)
    elif manifest.embedding_sidecars:
        raise MissingReferenceError(
            "snapshot declares embedding sidecars without a search index"
        )
    return manifest, prepared


def knowledge_record_envelopes(
    package: KnowledgeSnapshotPackage,
) -> tuple[Mapping[str, Any], ...]:
    """Return complete canonical record envelopes for retrieval and proof access."""

    if not isinstance(package, KnowledgeSnapshotPackage):
        raise TypeError("package must be a KnowledgeSnapshotPackage")
    return package.record_envelopes
