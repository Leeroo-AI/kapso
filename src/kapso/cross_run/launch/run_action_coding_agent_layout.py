"""Single structural path and environment ABI for native coding-agent actions."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

TRUSTED_WORKSPACE_PATH = "/kapso/workspace"
TEMPORARY_ROOT_PATH = "/kapso/tmp"
PROVIDER_WORKSPACE_PATH = "/kapso/tmp/provider-workspace"
PROVIDER_HOME_PATH = "/kapso/tmp/provider-home"
PROVIDER_CODEX_HOME_PATH = "/kapso/tmp/provider-home/.codex"
PROVIDER_CODEX_AUTH_PATH = "/kapso/tmp/provider-home/.codex/auth.json"
PROVIDER_CREDENTIAL_ROOT_PATH = "/kapso/credentials"
PROVIDER_CREDENTIAL_PATH = "/kapso/credentials/credentials"
PROVIDER_OUTPUT_PATH = "/kapso/tmp/provider-output"
PROVIDER_SUPPORT_PATH = "/kapso/tmp/provider-support"
PROVIDER_RESPONSE_SCHEMA_PATH = "/kapso/tmp/provider-support/response.schema.json"
PROVIDER_FINAL_PATH = "/kapso/tmp/provider-output/provider.final.json"
PROVIDER_MCP_CONFIGURATION_PATH = "/kapso/tmp/provider-support/mcp.config.json"
PROVIDER_PRIOR_KNOWLEDGE_PATH = "/kapso/tmp/provider-support/prior_knowledge.json"
PROVIDER_PRIOR_KNOWLEDGE_AUDIT_PATH = (
    "/kapso/tmp/provider-output/prior_knowledge.audit.jsonl"
)

_PROVIDER_ENVIRONMENT = MappingProxyType(
    {
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": PROVIDER_HOME_PATH,
        "CODEX_HOME": PROVIDER_CODEX_HOME_PATH,
        "LANG": "C",
        "LC_ALL": "C",
        "NO_COLOR": "1",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TERM": "dumb",
        "TMPDIR": PROVIDER_HOME_PATH,
    }
)


def coding_agent_provider_environment() -> Mapping[str, str]:
    """Return the provider's complete post-boundary process environment."""

    return _PROVIDER_ENVIRONMENT


__all__ = [
    "coding_agent_provider_environment",
    "PROVIDER_CODEX_AUTH_PATH",
    "PROVIDER_CODEX_HOME_PATH",
    "PROVIDER_CREDENTIAL_PATH",
    "PROVIDER_CREDENTIAL_ROOT_PATH",
    "PROVIDER_FINAL_PATH",
    "PROVIDER_HOME_PATH",
    "PROVIDER_MCP_CONFIGURATION_PATH",
    "PROVIDER_OUTPUT_PATH",
    "PROVIDER_PRIOR_KNOWLEDGE_AUDIT_PATH",
    "PROVIDER_PRIOR_KNOWLEDGE_PATH",
    "PROVIDER_RESPONSE_SCHEMA_PATH",
    "PROVIDER_SUPPORT_PATH",
    "PROVIDER_WORKSPACE_PATH",
    "TEMPORARY_ROOT_PATH",
    "TRUSTED_WORKSPACE_PATH",
]
