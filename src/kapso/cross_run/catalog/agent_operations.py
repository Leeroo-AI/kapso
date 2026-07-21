"""Shared provenance boundary for catalog coding-agent operations."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CodingAgentOperationReceipt,
    ContractValidationError,
    StrictContract,
)
from kapso.cross_run.settings import CatalogAgentSettings
from kapso.execution.coding_agents.structured_call import CodingAgentCallResult

_RECEIPT_ARTIFACT_FILENAMES = {
    "final.json",
    "invocation.json",
    "prompt.txt",
    "response_schema.json",
    "result.json",
    "stderr.txt",
    "stdout.txt",
}


class CatalogAgentOperationError(ValueError):
    """A catalog agent workspace or operation artifact set is invalid."""


@dataclass(frozen=True)
class CatalogAgentOperationRecord(StrictContract):
    """Exact model input/output binding behind framework-minted catalog facts."""

    operation_record_id: str
    operation_kind: str
    operation_receipt_id: str
    operation_preimage: Mapping[str, Any]
    final_output: str
    produced_object_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "catalog-agent-operation"
    IDENTITY_FIELD: ClassVar[str] = "operation_record_id"

    def _validate(self) -> None:
        if self.operation_kind not in {"claim_proposal", "catalog_review"}:
            raise ContractValidationError("catalog agent operation kind is invalid")
        require_content_id(self.operation_receipt_id, "operation_receipt_id")
        if not isinstance(self.operation_preimage, Mapping):
            raise ContractValidationError("operation preimage must be an object")
        if not isinstance(self.final_output, str) or not self.final_output.strip():
            raise ContractValidationError("catalog agent final output is empty")
        parse_json_bytes(self.final_output.encode("utf-8"))
        if self.produced_object_ids != tuple(sorted(set(self.produced_object_ids))):
            raise ContractValidationError(
                "produced object IDs must be sorted and unique"
            )
        for object_id in self.produced_object_ids:
            require_content_id(object_id, "produced_object_ids")

    @property
    def packet_payload(self) -> Mapping[str, Any]:
        packet = self.operation_preimage.get("packet")
        if not isinstance(packet, Mapping):
            raise CatalogAgentOperationError("operation preimage packet is absent")
        return packet

    def validate_receipt(self, receipt: CodingAgentOperationReceipt) -> None:
        if receipt.operation_receipt_id != self.operation_receipt_id:
            raise CatalogAgentOperationError("operation receipt identity differs")
        require_identifier(receipt.operation_id, "operation_id")
        if catalog_agent_operation_id(self.operation_preimage) != receipt.operation_id:
            raise CatalogAgentOperationError(
                "operation preimage does not match the receipt operation"
            )
        if tree_or_blob_digest(self.final_output.encode("utf-8")) != (
            receipt.artifact_checksums["final.json"]
        ):
            raise CatalogAgentOperationError(
                "operation final output does not match its receipt checksum"
            )


def catalog_agent_operation_id(preimage: Mapping[str, Any]) -> str:
    digest = tree_or_blob_digest(canonical_json_bytes(preimage))[7:]
    return f"agent_call_{digest[:32]}"


def validate_catalog_agent_workspace(workspace: Path) -> None:
    if not workspace.is_absolute() or not workspace.is_dir():
        raise CatalogAgentOperationError(
            "catalog agent workspace must be an existing absolute directory"
        )
    if workspace.is_symlink() or tuple(workspace.iterdir()):
        raise CatalogAgentOperationError(
            "catalog agent requires an empty, non-symlink workspace"
        )


def build_catalog_agent_operation_receipt(
    *,
    operation_id: str,
    principal_id: str,
    role: str,
    agent: CatalogAgentSettings,
    result: CodingAgentCallResult,
) -> tuple[CodingAgentOperationReceipt, str]:
    artifact_paths = tuple(Path(path) for path in result.artifacts)
    if not artifact_paths:
        raise CatalogAgentOperationError("catalog agent returned no artifacts")
    directories = {path.parent for path in artifact_paths}
    names = {path.name for path in artifact_paths}
    if len(directories) != 1 or names != _RECEIPT_ARTIFACT_FILENAMES - {"result.json"}:
        raise CatalogAgentOperationError("catalog agent artifact set is invalid")
    artifact_directory = next(iter(directories))
    complete_paths = artifact_paths + (artifact_directory / "result.json",)
    checksums: dict[str, str] = {}
    for path in complete_paths:
        status = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(status.st_mode):
            raise CatalogAgentOperationError(
                "catalog agent artifact must be a regular file"
            )
        checksums[path.name] = tree_or_blob_digest(path.read_bytes())
    final_output = (artifact_directory / "final.json").read_text(encoding="utf-8")
    final_payload = parse_json_bytes(final_output.encode("utf-8"))
    result_payload = parse_json_bytes(result.output)
    if canonical_json_bytes(final_payload) != canonical_json_bytes(result_payload):
        raise CatalogAgentOperationError(
            "catalog agent final artifact does not match result"
        )
    return (
        CodingAgentOperationReceipt.mint(
            operation_id=operation_id,
            principal_id=principal_id,
            role=role,
            cli=agent.cli,
            model=agent.model,
            effort=agent.effort,
            artifact_checksums=checksums,
        ),
        final_output,
    )
