"""Verified local access to one proof-closed prior-knowledge packet."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    freeze_json,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ContractValidationError,
    IdentityConflictError,
    MissingReferenceError,
    PriorKnowledgeSnapshot,
    StrictContract,
)
from kapso.cross_run.record_registry import (
    parse_knowledge_record_payload,
    record_identity,
)

_MATERIALIZATION_DIGEST_FIELDS = (
    "prior_knowledge_snapshot",
    "proof_records",
)
_CITABLE_RECORD_KINDS = frozenset(
    {"knowledge-claim-revision", "prior-idea", "transfer-episode"}
)
_CONTENT_TRUST_LABEL = "untrusted_prior_knowledge"
_INSTRUCTION_AUTHORITY_LABEL = "none"


class PriorKnowledgeAccessError(ValueError):
    """A local prior-knowledge materialization is invalid or inaccessible."""


def _require_record_envelope(
    value: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC) or set(value) != {
        "payload",
        "record_id",
        "record_kind",
    }:
        raise ContractValidationError(f"{name} must be a complete record envelope")
    record_id = require_content_id(value["record_id"], f"{name}.record_id")
    require_identifier(value["record_kind"], f"{name}.record_kind")
    payload = value["payload"]
    if not isinstance(payload, MappingABC) or not payload:
        raise ContractValidationError(f"{name}.payload must be a complete object")
    record_kind = value["record_kind"]
    record = parse_knowledge_record_payload(record_kind, payload)
    if record_identity(record) != record_id:
        raise IdentityConflictError(f"{name}.payload does not own its record ID")
    return freeze_json(value, name)


@dataclass(frozen=True)
class PriorKnowledgeAccessMaterialization(StrictContract):
    """One packet plus complete bytes for every non-selected proof record."""

    prior_knowledge_snapshot: PriorKnowledgeSnapshot
    proof_records: tuple[Mapping[str, Any], ...]
    materialization_digest: str

    def _validate(self) -> None:
        selected_records = tuple(
            _require_record_envelope(record, f"selected_records[{position}]")
            for position, record in enumerate(
                self.prior_knowledge_snapshot.selected_records
            )
        )
        selected_ids = tuple(record["record_id"] for record in selected_records)
        if selected_ids != self.prior_knowledge_snapshot.selected_record_ids:
            raise IdentityConflictError(
                "selected record envelopes do not match packet membership"
            )
        proof_records = tuple(
            _require_record_envelope(record, f"proof_records[{position}]")
            for position, record in enumerate(self.proof_records)
        )
        proof_ids = tuple(record["record_id"] for record in proof_records)
        if proof_ids != tuple(sorted(set(proof_ids))):
            raise ContractValidationError(
                "proof record envelopes must be sorted and unique"
            )
        expected_proof_ids = tuple(
            sorted(
                set(self.prior_knowledge_snapshot.proof_reference_ids)
                - set(selected_ids)
            )
        )
        if proof_ids != expected_proof_ids:
            raise MissingReferenceError(
                "proof record envelopes do not match the packet proof closure"
            )
        expected_digest = tree_or_blob_digest(
            canonical_json_bytes(
                {
                    field_name: getattr(self, field_name)
                    for field_name in _MATERIALIZATION_DIGEST_FIELDS
                }
            )
        )
        if self.materialization_digest != expected_digest:
            raise ContractValidationError(
                "prior-knowledge access materialization digest mismatch"
            )

    @classmethod
    def mint(
        cls,
        *,
        prior_knowledge_snapshot: PriorKnowledgeSnapshot,
        proof_records: tuple[Mapping[str, Any], ...],
    ) -> "PriorKnowledgeAccessMaterialization":
        payload = {
            "prior_knowledge_snapshot": prior_knowledge_snapshot,
            "proof_records": proof_records,
        }
        return cls(
            **payload,
            materialization_digest=tree_or_blob_digest(canonical_json_bytes(payload)),
        )

    def persist(self, destination: str | Path) -> Path:
        """Atomically create one immutable canonical packet file."""

        path = Path(destination)
        if not path.is_absolute() or ".." in path.parts or path != path.absolute():
            raise PriorKnowledgeAccessError(
                "prior-knowledge destination must be absolute and normalized"
            )
        if os.path.lexists(path):
            raise PriorKnowledgeAccessError(
                "prior-knowledge destination already exists"
            )
        parent = path.parent
        if parent.is_symlink() or not parent.is_dir():
            raise PriorKnowledgeAccessError(
                "prior-knowledge destination parent must be a real directory"
            )
        payload = self.to_json_bytes()
        with tempfile.TemporaryDirectory(
            prefix=".prior-knowledge-",
            dir=parent,
        ) as staging_directory:
            staged = Path(staging_directory) / "packet.json"
            descriptor = os.open(
                staged,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o444,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            verified = PriorKnowledgeAccess.open(
                staged,
                maximum_bytes=len(payload),
            )
            if verified.materialization != self:
                raise PriorKnowledgeAccessError(
                    "persisted prior-knowledge bytes changed before publication"
                )
            os.link(staged, path, follow_symlinks=False)
            parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            os.fsync(parent_descriptor)
            os.close(parent_descriptor)
        return path


class PriorKnowledgeAccess:
    """Read-only in-memory view over one verified local materialization."""

    def __init__(self, materialization: PriorKnowledgeAccessMaterialization):
        if not isinstance(materialization, PriorKnowledgeAccessMaterialization):
            raise TypeError(
                "materialization must be PriorKnowledgeAccessMaterialization"
            )
        self.materialization = materialization
        packet = materialization.prior_knowledge_snapshot
        records = (*packet.selected_records, *materialization.proof_records)
        self._records_by_id = {
            record["record_id"]: freeze_json(record, "prior knowledge record")
            for record in records
        }
        self._selected_ids = frozenset(packet.selected_record_ids)

    @classmethod
    def open(
        cls,
        materialization_path: str | Path,
        *,
        maximum_bytes: int,
    ) -> "PriorKnowledgeAccess":
        if (
            isinstance(maximum_bytes, bool)
            or not isinstance(maximum_bytes, int)
            or maximum_bytes <= 0
        ):
            raise ValueError(
                "prior-knowledge materialization byte budget must be positive"
            )
        path = Path(materialization_path)
        if not path.is_absolute() or ".." in path.parts:
            raise PriorKnowledgeAccessError(
                "prior-knowledge materialization path must be absolute and normalized"
            )
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            os.close(descriptor)
            raise PriorKnowledgeAccessError(
                "prior-knowledge materialization must be a regular file"
            )
        if status.st_size > maximum_bytes:
            os.close(descriptor)
            raise PriorKnowledgeAccessError(
                "prior-knowledge materialization exceeds its configured byte budget"
            )
        with os.fdopen(descriptor, "rb") as handle:
            payload = handle.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise PriorKnowledgeAccessError(
                "prior-knowledge materialization exceeds its configured byte budget"
            )
        materialization = PriorKnowledgeAccessMaterialization.from_json_bytes(payload)
        if payload != materialization.to_json_bytes():
            raise PriorKnowledgeAccessError(
                "prior-knowledge materialization bytes must be canonical"
            )
        return cls(materialization)

    @property
    def packet(self) -> PriorKnowledgeSnapshot:
        return self.materialization.prior_knowledge_snapshot

    def list_records(self) -> tuple[Mapping[str, Any], ...]:
        selection_metadata = self.packet.selection_metadata
        return tuple(
            {
                "record_id": record_id,
                "record_kind": self._records_by_id[record_id]["record_kind"],
                "membership": self.membership(record_id),
                "selection_metadata": selection_metadata.get(record_id),
            }
            for record_id in sorted(self._records_by_id)
        )

    def list_citable_records(self) -> tuple[Mapping[str, Any], ...]:
        """List scientific records that may be cited by local ideation."""

        return tuple(
            record
            for record in self.list_records()
            if record["record_kind"] in _CITABLE_RECORD_KINDS
        )

    def selection_metadata(self, record_id: str) -> Mapping[str, Any] | None:
        """Return persisted rank/compatibility data only for a selected root."""

        if record_id not in self._records_by_id:
            raise MissingReferenceError(
                "record is not a member of the persisted prior-knowledge packet"
            )
        return self.packet.selection_metadata.get(record_id)

    def get_record(self, record_id: str) -> Mapping[str, Any]:
        require_content_id(record_id, "prior knowledge record ID")
        record = self._records_by_id.get(record_id)
        if record is None:
            raise MissingReferenceError(
                "record is not a member of the persisted prior-knowledge packet"
            )
        return record

    def list_response_payload(self) -> Mapping[str, Any]:
        """Build the canonical gated response for packet discovery."""

        return self._response_payload({"records": self.list_records()})

    def record_response_payload(self, record_id: str) -> Mapping[str, Any]:
        """Build the canonical gated response for one complete packet member."""

        return self._response_payload(
            {
                "membership": self.membership(record_id),
                "record": self.get_record(record_id),
                "selection_metadata": self.selection_metadata(record_id),
            }
        )

    def _response_payload(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        packet = self.packet
        return freeze_json(
            {
                "security_labels": {
                    "content_trust": _CONTENT_TRUST_LABEL,
                    "instruction_authority": _INSTRUCTION_AUTHORITY_LABEL,
                },
                "provenance": {
                    "prior_knowledge_snapshot_id": (packet.prior_knowledge_snapshot_id),
                    "source_snapshot_id": packet.source_snapshot_id,
                    "task_context_binding_id": packet.task_context_binding_id,
                },
                **payload,
            },
            "prior knowledge response",
        )

    def membership(self, record_id: str) -> str:
        if record_id not in self._records_by_id:
            raise MissingReferenceError(
                "record is not a member of the persisted prior-knowledge packet"
            )
        return "selected" if record_id in self._selected_ids else "proof"
