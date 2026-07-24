"""Rebuildable metadata, lexical, and exact-vector snapshot indexes."""

from __future__ import annotations

import math
import re
import struct
import unicodedata
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Protocol

from kapso.cross_run.canonical import (
    CANONICALIZER_VERSION,
    canonical_json_bytes,
    freeze_json,
    parse_json_bytes,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CatalogEntryState,
    ContractValidationError,
    EmbeddingSidecar,
    KnowledgeSnapshotManifest,
    MissingReferenceError,
    StrictContract,
)
from kapso.cross_run.embedding_space import EmbeddingSpace

_INDEX_MANIFEST_PATH = "index/index-manifest.json"
_METADATA_PATH = "index/metadata.json"
_LEXICAL_PATH = "index/lexical.json"
_WORD_PATTERN = re.compile(r"[^\W_]+(?:[_-][^\W_]+)*", re.UNICODE)
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class SnapshotIndexError(ValueError):
    """A search index is stale, malformed, or incompatible with its snapshot."""


class PreparedSnapshotClosure(Protocol):
    """Minimum canonical snapshot closure needed to rebuild an index."""

    scope_contract_id: str
    catalog_generation_id: str
    record_closure_digest: str
    retrieval_root_ids: tuple[str, ...]
    entry_state_ids: tuple[str, ...]
    files: Mapping[str, bytes]
    scope_contract: Any
    catalog_generation: Any

    def record_by_id(self, record_id: str) -> Mapping[str, Any]: ...


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ContractValidationError(f"{name} must be a sha256 digest")


def _require_path(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise ContractValidationError(f"{name} must be an index/ path")
    path = PurePosixPath(value)
    if (
        not value.startswith("index/")
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ContractValidationError(f"{name} must be an index/ path")


@dataclass(frozen=True)
class EmbeddingVector:
    """One provider result bound to the complete canonical record input."""

    record_id: str
    input_digest: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        require_content_id(self.record_id, "embedding vector record_id")
        _require_digest(self.input_digest, "embedding vector input_digest")
        if (
            not isinstance(self.values, tuple)
            or not self.values
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in self.values
            )
        ):
            raise ContractValidationError(
                "embedding vector values must be finite floating-point numbers"
            )


@dataclass(frozen=True)
class EmbeddingVectorSet:
    """Complete vectors for every retrievable record in one embedding space."""

    space: EmbeddingSpace
    vectors: tuple[EmbeddingVector, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.vectors, tuple):
            raise ContractValidationError("embedding vectors must be a tuple")
        record_ids = tuple(vector.record_id for vector in self.vectors)
        if record_ids != tuple(sorted(set(record_ids))):
            raise ContractValidationError(
                "embedding vectors must be sorted and uniquely identified"
            )
        for vector in self.vectors:
            if len(vector.values) != self.space.dimensions:
                raise ContractValidationError("embedding vector dimension mismatch")


@dataclass(frozen=True)
class ExactVectorSidecarDescriptor(StrictContract):
    """Checksummed float32 vector sidecar pinned to canonical record inputs."""

    embedding_space: EmbeddingSpace
    record_ids: tuple[str, ...]
    ids_ref: str
    ids_checksum: str
    input_digests_ref: str
    input_digests_checksum: str
    data_ref: str
    data_checksum: str

    def _validate(self) -> None:
        if self.record_ids != tuple(sorted(set(self.record_ids))):
            raise ContractValidationError("vector record IDs must be sorted and unique")
        for record_id in self.record_ids:
            require_content_id(record_id, "vector record_ids")
        for name in ("ids_ref", "input_digests_ref", "data_ref"):
            _require_path(getattr(self, name), name)
        if len({self.ids_ref, self.input_digests_ref, self.data_ref}) != 3:
            raise ContractValidationError("vector sidecar paths must be distinct")
        for name in (
            "ids_checksum",
            "input_digests_checksum",
            "data_checksum",
        ):
            _require_digest(getattr(self, name), name)


@dataclass(frozen=True)
class SnapshotIndexManifest(StrictContract):
    """Exact rebuildable index identity, deliberately independent of snapshot ID."""

    index_manifest_id: str
    scope_contract_id: str
    catalog_generation_id: str
    catalog_generation: int
    record_closure_digest: str
    canonicalizer_version: str
    indexed_record_ids: tuple[str, ...]
    metadata_ref: str
    metadata_checksum: str
    lexical_ref: str
    lexical_checksum: str
    vector_sidecars: tuple[ExactVectorSidecarDescriptor, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "snapshot-index-manifest"
    IDENTITY_FIELD: ClassVar[str] = "index_manifest_id"

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "index scope_contract_id")
        require_content_id(
            self.catalog_generation_id,
            "index catalog_generation_id",
        )
        if self.catalog_generation < 0:
            raise ContractValidationError(
                "index catalog generation must be non-negative"
            )
        _require_digest(self.record_closure_digest, "index record_closure_digest")
        require_identifier(self.canonicalizer_version, "canonicalizer_version")
        if self.indexed_record_ids != tuple(sorted(set(self.indexed_record_ids))):
            raise ContractValidationError(
                "indexed record IDs must be sorted and unique"
            )
        for record_id in self.indexed_record_ids:
            require_content_id(record_id, "indexed_record_ids")
        for name in ("metadata_ref", "lexical_ref"):
            _require_path(getattr(self, name), name)
        if self.metadata_ref == self.lexical_ref:
            raise ContractValidationError("metadata and lexical paths must differ")
        _require_digest(self.metadata_checksum, "metadata_checksum")
        _require_digest(self.lexical_checksum, "lexical_checksum")
        spaces = tuple(
            sidecar.embedding_space.embedding_space_id
            for sidecar in self.vector_sidecars
        )
        if spaces != tuple(sorted(set(spaces))):
            raise ContractValidationError(
                "index vector sidecars must be sorted by unique embedding space"
            )
        if any(
            sidecar.record_ids != self.indexed_record_ids
            for sidecar in self.vector_sidecars
        ):
            raise MissingReferenceError(
                "each vector sidecar must cover every indexed record"
            )


@dataclass(frozen=True)
class SnapshotSearchIndex:
    """Verified local query primitives with no admission or retrieval policy."""

    manifest: SnapshotIndexManifest
    metadata_by_id: Mapping[str, Mapping[str, Any]]
    files: Mapping[str, bytes]
    _lexical_postings: Mapping[str, tuple[str, ...]]
    _vectors_by_space: Mapping[str, Mapping[str, tuple[float, ...]]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata_by_id",
            MappingProxyType(
                {
                    record_id: freeze_json(metadata, "index metadata")
                    for record_id, metadata in sorted(self.metadata_by_id.items())
                }
            ),
        )
        object.__setattr__(
            self,
            "files",
            MappingProxyType(
                {path: payload for path, payload in sorted(self.files.items())}
            ),
        )
        object.__setattr__(
            self,
            "_lexical_postings",
            MappingProxyType(
                {
                    term: tuple(record_ids)
                    for term, record_ids in sorted(self._lexical_postings.items())
                }
            ),
        )
        object.__setattr__(
            self,
            "_vectors_by_space",
            MappingProxyType(
                {
                    space_id: MappingProxyType(dict(vectors))
                    for space_id, vectors in sorted(self._vectors_by_space.items())
                }
            ),
        )

    @property
    def record_closure_digest(self) -> str:
        return self.manifest.record_closure_digest

    @property
    def embedding_sidecars(self) -> tuple[EmbeddingSidecar, ...]:
        return tuple(
            EmbeddingSidecar(
                embedding_space_id=sidecar.embedding_space.embedding_space_id,
                asset_ref=sidecar.data_ref,
                checksum=sidecar.data_checksum,
            )
            for sidecar in self.manifest.vector_sidecars
        )

    def lexical_scores(self, query_text: str) -> Mapping[str, float]:
        """Return deterministic exact-term overlap scores for matching records."""

        if not isinstance(query_text, str) or not query_text.strip():
            raise SnapshotIndexError("lexical query must be non-empty text")
        query_terms = _lexical_terms(query_text)
        matched_counts: dict[str, int] = {}
        for term in sorted(query_terms):
            for record_id in self._lexical_postings.get(term, ()):
                matched_counts[record_id] = matched_counts.get(record_id, 0) + 1
        scores = {
            record_id: matched_count / len(query_terms)
            for record_id, matched_count in matched_counts.items()
        }
        return MappingProxyType(
            dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))
        )

    def semantic_scores(
        self,
        query_vector: tuple[float, ...],
        embedding_space_id: str,
    ) -> Mapping[str, float]:
        """Return exact cosine scores without crossing embedding spaces."""

        require_content_id(embedding_space_id, "embedding_space_id")
        descriptor = next(
            (
                sidecar
                for sidecar in self.manifest.vector_sidecars
                if sidecar.embedding_space.embedding_space_id == embedding_space_id
            ),
            None,
        )
        if descriptor is None:
            raise SnapshotIndexError("embedding space is absent from this index")
        if (
            not isinstance(query_vector, tuple)
            or len(query_vector) != descriptor.embedding_space.dimensions
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in query_vector
            )
        ):
            raise SnapshotIndexError("semantic query vector is invalid")
        query_norm = math.sqrt(math.fsum(value * value for value in query_vector))
        if query_norm == 0.0:
            raise SnapshotIndexError("semantic query vector has zero norm")
        scores: dict[str, float] = {}
        for record_id, vector in self._vectors_by_space[embedding_space_id].items():
            vector_norm = math.sqrt(math.fsum(value * value for value in vector))
            scores[record_id] = math.fsum(
                left * right for left, right in zip(query_vector, vector)
            ) / (query_norm * vector_norm)
        return MappingProxyType(
            dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))
        )

    def verify(self, snapshot_manifest: KnowledgeSnapshotManifest) -> None:
        """Bind these sidecar bytes to one finalized snapshot manifest."""

        if not isinstance(snapshot_manifest, KnowledgeSnapshotManifest):
            raise TypeError("snapshot_manifest must be KnowledgeSnapshotManifest")
        if (
            snapshot_manifest.scope_contract_id != self.manifest.scope_contract_id
            or snapshot_manifest.catalog_generation != self.manifest.catalog_generation
            or snapshot_manifest.proof_dependency_closure_ids
            != tuple(sorted(snapshot_manifest.proof_dependency_closure_ids))
        ):
            raise SnapshotIndexError("index belongs to another snapshot closure")
        for path, payload in self.files.items():
            if snapshot_manifest.checksums.get(path) != tree_or_blob_digest(payload):
                raise SnapshotIndexError(
                    f"snapshot does not bind index sidecar: {path}"
                )
        if snapshot_manifest.embedding_sidecars != self.embedding_sidecars:
            raise SnapshotIndexError(
                "snapshot embedding descriptors differ from the index"
            )

    @classmethod
    def build(
        cls,
        prepared: PreparedSnapshotClosure,
        vector_sets: tuple[EmbeddingVectorSet, ...] = (),
    ) -> SnapshotSearchIndex:
        """Build deterministic sidecars from canonical record envelopes."""

        metadata = _build_metadata(prepared)
        lexical = _build_lexical(prepared)
        metadata_bytes = canonical_json_bytes(
            tuple(metadata[record_id] for record_id in sorted(metadata))
        )
        lexical_bytes = canonical_json_bytes(lexical)
        files: dict[str, bytes] = {
            _METADATA_PATH: metadata_bytes,
            _LEXICAL_PATH: lexical_bytes,
        }
        expected_record_ids = prepared.retrieval_root_ids
        if not isinstance(vector_sets, tuple):
            raise SnapshotIndexError("vector_sets must be a tuple")
        if vector_sets and not expected_record_ids:
            raise SnapshotIndexError("an EMPTY index cannot publish embedding sidecars")
        spaces = tuple(
            vector_set.space.embedding_space_id for vector_set in vector_sets
        )
        if spaces != tuple(sorted(set(spaces))):
            raise SnapshotIndexError(
                "vector sets must be sorted by unique embedding space"
            )
        vector_descriptors: list[ExactVectorSidecarDescriptor] = []
        for vector_set in vector_sets:
            vector_ids = tuple(vector.record_id for vector in vector_set.vectors)
            if vector_ids != expected_record_ids:
                raise MissingReferenceError(
                    "vector set does not cover the complete retrievable record set"
                )
            input_digests = tuple(
                tree_or_blob_digest(
                    canonical_json_bytes(prepared.record_by_id(record_id))
                )
                for record_id in vector_ids
            )
            if tuple(vector.input_digest for vector in vector_set.vectors) != (
                input_digests
            ):
                raise SnapshotIndexError(
                    "vector input digest differs from canonical record text"
                )
            vector_data = _float32_bytes(vector_set)
            space_digest = vector_set.space.embedding_space_id.rsplit(":", 1)[1]
            base_path = f"index/vectors/{space_digest}"
            ids_ref = f"{base_path}/ids.json"
            inputs_ref = f"{base_path}/input-digests.json"
            data_ref = f"{base_path}/vectors.f32"
            ids_bytes = canonical_json_bytes(vector_ids)
            inputs_bytes = canonical_json_bytes(input_digests)
            files[ids_ref] = ids_bytes
            files[inputs_ref] = inputs_bytes
            files[data_ref] = vector_data
            vector_descriptors.append(
                ExactVectorSidecarDescriptor(
                    embedding_space=vector_set.space,
                    record_ids=vector_ids,
                    ids_ref=ids_ref,
                    ids_checksum=tree_or_blob_digest(ids_bytes),
                    input_digests_ref=inputs_ref,
                    input_digests_checksum=tree_or_blob_digest(inputs_bytes),
                    data_ref=data_ref,
                    data_checksum=tree_or_blob_digest(vector_data),
                )
            )
        manifest = SnapshotIndexManifest.mint(
            scope_contract_id=prepared.scope_contract_id,
            catalog_generation_id=prepared.catalog_generation_id,
            catalog_generation=prepared.catalog_generation.generation_number,
            record_closure_digest=prepared.record_closure_digest,
            canonicalizer_version=CANONICALIZER_VERSION,
            indexed_record_ids=expected_record_ids,
            metadata_ref=_METADATA_PATH,
            metadata_checksum=tree_or_blob_digest(metadata_bytes),
            lexical_ref=_LEXICAL_PATH,
            lexical_checksum=tree_or_blob_digest(lexical_bytes),
            vector_sidecars=tuple(vector_descriptors),
        )
        files[_INDEX_MANIFEST_PATH] = manifest.to_json_bytes()
        return cls.open(prepared, files)

    @classmethod
    def open(
        cls,
        prepared: PreparedSnapshotClosure,
        files: Mapping[str, bytes],
    ) -> SnapshotSearchIndex:
        """Verify and open independently rebuildable index sidecars."""

        if not isinstance(files, MappingABC) or any(
            not isinstance(path, str) or not isinstance(payload, bytes)
            for path, payload in files.items()
        ):
            raise SnapshotIndexError("index files must map paths to bytes")
        manifest_bytes = files.get(_INDEX_MANIFEST_PATH)
        if manifest_bytes is None:
            raise MissingReferenceError("index manifest is absent")
        manifest = SnapshotIndexManifest.from_json_bytes(manifest_bytes)
        if manifest_bytes != manifest.to_json_bytes():
            raise SnapshotIndexError("index manifest bytes are not canonical")
        if (
            manifest.scope_contract_id != prepared.scope_contract_id
            or manifest.catalog_generation_id != prepared.catalog_generation_id
            or manifest.catalog_generation
            != prepared.catalog_generation.generation_number
            or manifest.record_closure_digest != prepared.record_closure_digest
            or manifest.canonicalizer_version != CANONICALIZER_VERSION
            or manifest.indexed_record_ids != prepared.retrieval_root_ids
        ):
            raise SnapshotIndexError("index is stale for the prepared snapshot")
        expected_files = {
            _INDEX_MANIFEST_PATH,
            manifest.metadata_ref,
            manifest.lexical_ref,
            *(
                path
                for sidecar in manifest.vector_sidecars
                for path in (
                    sidecar.ids_ref,
                    sidecar.input_digests_ref,
                    sidecar.data_ref,
                )
            ),
        }
        if set(files) != expected_files:
            raise MissingReferenceError("index file closure is incomplete")
        checksum_by_path = {
            manifest.metadata_ref: manifest.metadata_checksum,
            manifest.lexical_ref: manifest.lexical_checksum,
        }
        for sidecar in manifest.vector_sidecars:
            checksum_by_path.update(
                {
                    sidecar.ids_ref: sidecar.ids_checksum,
                    sidecar.input_digests_ref: sidecar.input_digests_checksum,
                    sidecar.data_ref: sidecar.data_checksum,
                }
            )
        for path, expected_checksum in checksum_by_path.items():
            if tree_or_blob_digest(files[path]) != expected_checksum:
                raise SnapshotIndexError(f"index sidecar checksum mismatch: {path}")
        expected_metadata = _build_metadata(prepared)
        expected_lexical = _build_lexical(prepared)
        if files[manifest.metadata_ref] != canonical_json_bytes(
            tuple(
                expected_metadata[record_id] for record_id in sorted(expected_metadata)
            )
        ) or files[manifest.lexical_ref] != canonical_json_bytes(expected_lexical):
            raise SnapshotIndexError(
                "metadata or lexical index differs from canonical records"
            )
        parsed_metadata = parse_json_bytes(files[manifest.metadata_ref])
        parsed_lexical = parse_json_bytes(files[manifest.lexical_ref])
        if not isinstance(parsed_metadata, list) or not isinstance(
            parsed_lexical, MappingABC
        ):
            raise SnapshotIndexError("metadata or lexical index shape is invalid")
        metadata_by_id = {
            entry["record_id"]: entry
            for entry in parsed_metadata
            if isinstance(entry, MappingABC) and "record_id" in entry
        }
        if len(metadata_by_id) != len(parsed_metadata):
            raise SnapshotIndexError("metadata index contains invalid records")
        lexical_postings = {
            term: tuple(record_ids)
            for term, record_ids in parsed_lexical.items()
            if isinstance(term, str) and isinstance(record_ids, list)
        }
        if len(lexical_postings) != len(parsed_lexical):
            raise SnapshotIndexError("lexical index contains invalid postings")
        vectors_by_space: dict[str, Mapping[str, tuple[float, ...]]] = {}
        for sidecar in manifest.vector_sidecars:
            parsed_ids = parse_json_bytes(files[sidecar.ids_ref])
            parsed_input_digests = parse_json_bytes(files[sidecar.input_digests_ref])
            if (
                not isinstance(parsed_ids, list)
                or tuple(parsed_ids) != sidecar.record_ids
                or not isinstance(parsed_input_digests, list)
            ):
                raise SnapshotIndexError("vector identity sidecar is malformed")
            expected_input_digests = tuple(
                tree_or_blob_digest(
                    canonical_json_bytes(prepared.record_by_id(record_id))
                )
                for record_id in sidecar.record_ids
            )
            if tuple(parsed_input_digests) != expected_input_digests:
                raise SnapshotIndexError("vector input sidecar is stale")
            vectors = _read_float32_vectors(
                files[sidecar.data_ref],
                sidecar.record_ids,
                sidecar.embedding_space.dimensions,
            )
            vectors_by_space[sidecar.embedding_space.embedding_space_id] = vectors
        return cls(
            manifest=manifest,
            metadata_by_id=metadata_by_id,
            files=files,
            _lexical_postings=lexical_postings,
            _vectors_by_space=vectors_by_space,
        )


def _lexical_terms(value: Any) -> frozenset[str]:
    terms: set[str] = set()

    def visit(child: Any) -> None:
        if isinstance(child, MappingABC):
            for key in sorted(child):
                visit(key)
                visit(child[key])
        elif isinstance(child, (list, tuple)):
            for item in child:
                visit(item)
        elif isinstance(child, str):
            normalized = unicodedata.normalize("NFKC", child).casefold().strip()
            if normalized:
                terms.add(f"={normalized}")
                terms.update(
                    f"w:{match.group(0)}"
                    for match in _WORD_PATTERN.finditer(normalized)
                )

    visit(value)
    return frozenset(terms)


def _build_lexical(prepared: PreparedSnapshotClosure) -> Mapping[str, tuple[str, ...]]:
    postings: dict[str, list[str]] = {}
    for record_id in prepared.retrieval_root_ids:
        for term in sorted(_lexical_terms(prepared.record_by_id(record_id))):
            postings.setdefault(term, []).append(record_id)
    return MappingProxyType(
        {term: tuple(record_ids) for term, record_ids in sorted(postings.items())}
    )


def _build_metadata(
    prepared: PreparedSnapshotClosure,
) -> Mapping[str, Mapping[str, Any]]:
    states_by_subject = {
        state.subject_payload_id: state
        for state in (
            CatalogEntryState.from_json_bytes(
                prepared.files[_package_record_path(state_id)]
            )
            for state_id in prepared.entry_state_ids
        )
    }
    metadata: dict[str, Mapping[str, Any]] = {}
    for record_id in prepared.retrieval_root_ids:
        envelope = prepared.record_by_id(record_id)
        payload = envelope["payload"]
        record_kind = envelope["record_kind"]
        context = payload.get("task_context_binding", {})
        if not isinstance(context, MappingABC):
            context = {}
        source = payload.get("source", {})
        if not isinstance(source, MappingABC):
            source = {}
        evaluation_ids: tuple[str, ...] = ()
        timestamps: tuple[str, ...] = ()
        outcome = "frontier"
        if record_kind == "transfer-episode":
            attempts = payload["attempts"]
            terminal = attempts[payload["terminal_attempt_revision"]]
            evaluation_ids = tuple(
                sorted(
                    fingerprint["evaluation_fingerprint_id"]
                    for fingerprint in terminal["evaluation_fingerprints"]
                )
            )
            timestamps = (terminal["captured_at"],)
            effect = terminal["source_parent_effect"]
            if terminal["execution_status"] != "completed":
                outcome = "frontier"
            elif effect is None or terminal["evaluation_status"] != "valid":
                outcome = "inconclusive"
            elif effect["normalized_delta"] > 0.0:
                outcome = "positive"
            elif effect["normalized_delta"] < 0.0:
                outcome = "negative"
            else:
                outcome = "inconclusive"
        elif record_kind == "knowledge-claim-revision":
            outcome = "inconclusive"
        descriptor = payload.get("descriptor", {})
        if not isinstance(descriptor, MappingABC):
            descriptor = {}
        lineage_ids = tuple(
            sorted(
                value
                for value in (
                    payload.get("supersedes_projection_id"),
                    payload.get("parent_episode_ref"),
                    *payload.get("supersedes_revision_ids", ()),
                )
                if value is not None
            )
        )
        state = states_by_subject[record_id]
        metadata[record_id] = MappingProxyType(
            {
                "applicability": payload.get(
                    "applicability_predicates",
                    context.get("transfer_dimensions", {}),
                ),
                "approach_family": descriptor.get("approach_family"),
                "context_dimensions": context.get("transfer_dimensions", {}),
                "evaluation_fingerprint_ids": evaluation_ids,
                "exclusions": payload.get("explicit_exclusions", ()),
                "lineage_ids": lineage_ids,
                "mechanism": payload.get("mechanism", descriptor.get("mechanism")),
                "outcome": outcome,
                "record_id": record_id,
                "record_kind": record_kind,
                "run_id": source.get("run_id"),
                "scope_contract_id": prepared.scope_contract_id,
                "scope_id": context.get("scope_id", prepared.scope_contract.scope_id),
                "task_context_binding_id": context.get("task_context_binding_id"),
                "task_family_id": context.get("task_family_id"),
                "timestamps": timestamps,
                "trust_state": state.admission_state.value,
            }
        )
    return MappingProxyType(metadata)


def _float32_bytes(vector_set: EmbeddingVectorSet) -> bytes:
    values: list[float] = []
    for vector in vector_set.vectors:
        for value in vector.values:
            quantized = struct.unpack("<f", struct.pack("<f", value))[0]
            if not math.isfinite(quantized):
                raise SnapshotIndexError("embedding value overflows float32")
            values.append(quantized)
    if values and all(value == 0.0 for value in values):
        raise SnapshotIndexError("vector sidecar cannot contain only zero vectors")
    return struct.pack(f"<{len(values)}f", *values)


def _read_float32_vectors(
    payload: bytes,
    record_ids: tuple[str, ...],
    dimensions: int,
) -> Mapping[str, tuple[float, ...]]:
    expected_values = len(record_ids) * dimensions
    if len(payload) != expected_values * 4:
        raise SnapshotIndexError("float32 vector sidecar has the wrong size")
    values = struct.unpack(f"<{expected_values}f", payload)
    vectors: dict[str, tuple[float, ...]] = {}
    for position, record_id in enumerate(record_ids):
        start = position * dimensions
        vector = tuple(values[start : start + dimensions])
        if not vector or any(not math.isfinite(value) for value in vector):
            raise SnapshotIndexError("float32 vector sidecar is malformed")
        if math.fsum(value * value for value in vector) == 0.0:
            raise SnapshotIndexError("float32 vector has zero norm")
        vectors[record_id] = vector
    return MappingProxyType(vectors)


def _package_record_path(record_id: str) -> str:
    namespace, digest = record_id.split(":sha256:", 1)
    return f"records/{namespace}/{digest}.json"
