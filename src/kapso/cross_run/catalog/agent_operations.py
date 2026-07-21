"""Shared provenance boundary for catalog coding-agent operations."""

from __future__ import annotations

import stat
from pathlib import Path

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import CodingAgentOperationReceipt
from kapso.cross_run.record_contracts import CatalogAgentOperationError
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
