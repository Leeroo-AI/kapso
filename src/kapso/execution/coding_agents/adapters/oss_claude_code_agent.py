"""OSS Claude Code adapter — the Claude Code CLI driven by an open model
served through an Anthropic-compatible endpoint (e.g. GLM 5.2 on Fireworks).

Identical session mechanics to ClaudeCodeCodingAgent (same CLI, tools,
streaming, timeouts, env_strip/env_defaults); only authentication differs.
Required ``agent_specific`` keys:

- ``base_url``: the Anthropic-compatible endpoint,
  e.g. ``https://api.fireworks.ai/inference``.
- ``auth_token_env``: the NAME of the env var holding the provider key
  (e.g. ``FIREWORKS_API_KEY``) — the secret itself never appears in config.

The CLI child env gets ``ANTHROPIC_BASE_URL`` + ``ANTHROPIC_AUTH_TOKEN`` and
has every first-party credential and model-slot override stripped, so the CLI
can neither silently fall back to Anthropic auth nor have its model remapped
by ambient ``ANTHROPIC_*_MODEL`` vars. Model-slot vars for the CLI's internal
small/fast and subagent calls are pinned to the member's own model — the
endpoint serves no Anthropic model ids.

Probe-verified against Fireworks GLM 5.2 (2026-07-21): model id
``accounts/fireworks/models/glm-5p2``; bearer auth via ANTHROPIC_AUTH_TOKEN;
``--effort max`` flows through to GLM's native thinking (larger budgets =
measurably longer reasoning); Read/Bash agentic tool loop works.
"""

import shutil
from typing import Dict

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.adapters.claude_code_agent import (
    ClaudeCodeCodingAgent,
)

# First-party credentials and ambient model-slot overrides that must never
# reach the child: any of these would let the CLI bypass the OSS endpoint.
FIRST_PARTY_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_PROFILE",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
)

# CLI internal slots pinned to the member's own model: the endpoint cannot
# serve the Anthropic ids the CLI would otherwise route these calls to.
PINNED_MODEL_SLOTS = ("ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL")


class OssClaudeCodeCodingAgent(ClaudeCodeCodingAgent):
    """Claude Code CLI against a custom Anthropic-compatible endpoint."""

    def __init__(self, config: CodingAgentConfig):
        base_url = config.agent_specific.get("base_url")
        auth_token_env = config.agent_specific.get("auth_token_env")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(
                "oss_claude_code requires agent_specific.base_url "
                "(an Anthropic-compatible endpoint URL)"
            )
        if not isinstance(auth_token_env, str) or not auth_token_env.strip():
            raise ValueError(
                "oss_claude_code requires agent_specific.auth_token_env "
                "(the NAME of the env var holding the provider key)"
            )
        # Set before super().__init__: the parent constructor calls
        # _verify_cli(), which the overrides below rely on.
        self._base_url = base_url.strip()
        self._auth_token_env = auth_token_env.strip()
        super().__init__(config)

    def _verify_cli(self):
        """Verify the CLI exists and the provider key is available.

        Replaces the parent's first-party auth resolution entirely: this
        adapter has exactly one auth mode (the configured endpoint).
        """
        if not shutil.which("claude"):
            raise RuntimeError(
                "Claude Code CLI not found. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )
        env = self._get_effective_env()
        if not env.get(self._auth_token_env):
            raise ValueError(
                f"{self._auth_token_env} not set — oss_claude_code needs it "
                f"to authenticate against {self._base_url}"
            )
        self._auth_mode = "endpoint"
        self._use_bedrock = False

    def _get_env(self) -> Dict[str, str]:
        """Child env: endpoint auth only, first-party escape hatches removed."""
        env = self._get_effective_env()
        token = env.get(self._auth_token_env)
        if not token:
            raise ValueError(
                f"{self._auth_token_env} not set — cannot start an "
                "oss_claude_code session"
            )
        self._remove_provider_flags(env)
        for name in FIRST_PARTY_ENV_VARS:
            env.pop(name, None)
        env["ANTHROPIC_BASE_URL"] = self._base_url
        env["ANTHROPIC_AUTH_TOKEN"] = token
        for name in PINNED_MODEL_SLOTS:
            env[name] = self.config.model
        for name in self._env_strip:
            env.pop(name, None)
        for name, value in self._env_defaults.items():
            env.setdefault(name, value)
        return env
