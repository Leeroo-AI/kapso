"""Canonical durable artifact names for structured coding-agent operations."""

from enum import Enum


class CodingAgentWorkspaceAccess(str, Enum):
    READ_ONLY = "read_only"
    EDIT_WORKSPACE = "edit_workspace"


CODING_AGENT_INPUT_ARTIFACT_FILENAMES = (
    "prompt.txt",
    "response_schema.json",
    "invocation.json",
    "prior_knowledge.json",
    "mcp_config.json",
)
CODING_AGENT_COMMON_OUTPUT_ARTIFACT_FILENAMES = (
    "stdout.txt",
    "stderr.txt",
    "final.json",
    "mcp_audit.jsonl",
)
CODING_AGENT_RESULT_FILENAME = "result.json"
CODING_AGENT_WORKSPACE_DELTA_FILENAME = "workspace-delta.json"


def coding_agent_output_artifact_filenames(
    workspace_access: CodingAgentWorkspaceAccess,
) -> tuple[str, ...]:
    if not isinstance(workspace_access, CodingAgentWorkspaceAccess):
        raise ValueError("coding-agent access mode is invalid")
    if workspace_access is CodingAgentWorkspaceAccess.EDIT_WORKSPACE:
        return (
            *CODING_AGENT_COMMON_OUTPUT_ARTIFACT_FILENAMES,
            CODING_AGENT_WORKSPACE_DELTA_FILENAME,
        )
    return CODING_AGENT_COMMON_OUTPUT_ARTIFACT_FILENAMES


def coding_agent_returned_artifact_filenames(
    workspace_access: CodingAgentWorkspaceAccess,
) -> tuple[str, ...]:
    return (
        CODING_AGENT_INPUT_ARTIFACT_FILENAMES
        + coding_agent_output_artifact_filenames(workspace_access)
    )


def coding_agent_artifact_filenames(
    workspace_access: CodingAgentWorkspaceAccess,
) -> tuple[str, ...]:
    return (
        *coding_agent_returned_artifact_filenames(workspace_access),
        CODING_AGENT_RESULT_FILENAME,
    )
