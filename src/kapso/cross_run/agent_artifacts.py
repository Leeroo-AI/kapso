"""Canonical durable artifact names for structured coding-agent operations."""

CODING_AGENT_INPUT_ARTIFACT_FILENAMES = (
    "prompt.txt",
    "response_schema.json",
    "invocation.json",
    "prior_knowledge.json",
    "mcp_config.json",
)
CODING_AGENT_OUTPUT_ARTIFACT_FILENAMES = (
    "stdout.txt",
    "stderr.txt",
    "final.json",
    "mcp_audit.jsonl",
)
CODING_AGENT_RESULT_FILENAME = "result.json"
CODING_AGENT_RETURNED_ARTIFACT_FILENAMES = (
    CODING_AGENT_INPUT_ARTIFACT_FILENAMES + CODING_AGENT_OUTPUT_ARTIFACT_FILENAMES
)
CODING_AGENT_ARTIFACT_FILENAMES = (
    *CODING_AGENT_RETURNED_ARTIFACT_FILENAMES,
    CODING_AGENT_RESULT_FILENAME,
)
