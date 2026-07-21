"""Minimal inherited environment for one coding-agent provider process."""

from __future__ import annotations

import os

_BASE_KEYS = frozenset(
    {
        "ALL_PROXY",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LC_ALL",
        "NO_PROXY",
        "PATH",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TMPDIR",
        "USER",
    }
)
_CODEX_AUTH_KEYS = frozenset({"CODEX_HOME"})
_CLAUDE_AUTH_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BEDROCK_BASE_URL",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "AWS_ACCESS_KEY_ID",
        "AWS_CONFIG_FILE",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLOUD_ML_PROJECT_ID",
        "CLOUD_ML_REGION",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
)


def coding_agent_credential_environment(cli: str) -> dict[str, str]:
    """Broker only process/runtime keys and the selected CLI's auth keys."""

    if cli not in {"codex", "claude_code"}:
        raise ValueError("coding-agent credential environment CLI is invalid")
    allowed = set(_BASE_KEYS)
    allowed.update(_CODEX_AUTH_KEYS if cli == "codex" else _CLAUDE_AUTH_KEYS)
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    if "PATH" not in environment:
        raise ValueError("coding-agent subprocess requires an explicit PATH")
    return environment
