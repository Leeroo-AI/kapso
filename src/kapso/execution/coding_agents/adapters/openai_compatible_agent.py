"""Text-only inference through OpenAI-compatible Chat Completions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from kapso.core.api_endpoint import (
    openai_compatible_options,
    validate_api_base_url,
)
from kapso.execution.coding_agents.base import (
    CodingAgentConfig,
    CodingAgentInterface,
    CodingResult,
)


class OpenAICompatibleInferenceAgent(CodingAgentInterface):
    """Serve CliInference through the shared agent interface, without tools.

    Defaults live in agents.yaml. The endpoint comes only from config;
    api_key_env names the credential, never an endpoint setting.
    """

    def __init__(self, config: CodingAgentConfig):
        super().__init__(config)
        spec = openai_compatible_options(config.agent_specific)
        self.workspace: Optional[Path] = None
        self._base_url = validate_api_base_url(spec["base_url"])
        self._max_output_tokens = int(spec["max_output_tokens"])
        temperature = spec["temperature"]
        self._temperature = None if temperature is None else float(temperature)
        self._timeout = float(spec["timeout"])
        self._request_options = dict(spec.get("request_options") or {})
        reserved = {
            "model",
            "messages",
            "timeout",
            "stream",
            "n",
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "extra_body",
            "extra_headers",
            "extra_query",
        }
        if reserved.intersection(self._request_options):
            raise ValueError(
                "request_options cannot override session fields: "
                + ", ".join(
                    sorted(reserved.intersection(self._request_options))
                )
            )
        if self._timeout <= 0 or self._max_output_tokens <= 0:
            raise ValueError("timeout and max_output_tokens must be positive")
        key_env = spec["api_key_env"]
        if not isinstance(key_env, str) or not key_env.strip():
            raise ValueError("api_key_env must name an environment variable")
        api_key = os.getenv(key_env.strip())
        if not api_key:
            if not spec["allow_missing_api_key"]:
                raise ValueError(
                    f"{key_env} not set — openai_compatible needs it"
                )
            api_key = "not-needed"
        # Always pass the endpoint: the SDK otherwise reads OPENAI_BASE_URL.
        self._client = OpenAI(
            api_key=api_key, base_url=self._base_url, max_retries=0
        )

    def initialize(self, workspace: str) -> None:
        self.workspace = Path(workspace).resolve()

    def generate_code(
        self,
        prompt: str,
        debug_mode: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> CodingResult:
        """Return text through the shared interface; never apply file edits."""
        if self.workspace is None:
            raise RuntimeError(
                "Agent not initialized. Call initialize() first."
            )
        model = self.config.debug_model if debug_mode else self.config.model
        options: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the user's request directly in the requested "
                        "format. You have no filesystem, shell, or web-search "
                        "tools. Do not claim to have read files, run code, "
                        "or searched the web."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self._max_output_tokens,
            "timeout": (
                self._timeout if timeout_seconds is None else timeout_seconds
            ),
        }
        if self._temperature is not None:
            options["temperature"] = self._temperature
        options.update(self._request_options)
        if "max_completion_tokens" in options:
            options.pop("max_tokens", None)
        response = self._client.chat.completions.create(**options)
        if not response.choices:
            return CodingResult(
                success=False, output="", error="No completion choices"
            )
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) not in (None, "stop"):
            return CodingResult(
                success=False,
                output="",
                error=(
                    "Incomplete completion: "
                    f"finish_reason={choice.finish_reason}"
                ),
            )
        if getattr(choice.message, "tool_calls", None):
            return CodingResult(
                success=False, output="", error="Tool calls are not supported"
            )
        output = choice.message.content or ""
        if not output.strip():
            return CodingResult(
                success=False,
                output="",
                error="Endpoint returned empty output",
            )
        return CodingResult(
            success=True,
            output=output,
            files_changed=[],
            metadata={
                "model": model,
                "base_url": self._base_url,
                "usage": self._usage_dict(getattr(response, "usage", None)),
            },
        )

    @staticmethod
    def _usage_dict(usage: Any) -> Dict[str, int]:
        if usage is None:
            return {}
        return {
            name: int(getattr(usage, name, 0) or 0)
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        }

    def cleanup(self) -> None:
        self._client.close()
        self.workspace = None

    def get_capabilities(self) -> Dict[str, bool]:
        return {
            "native_git": False,
            "sandbox": False,
            "planning_mode": False,
            "cost_tracking": False,
            "streaming": False,
        }
