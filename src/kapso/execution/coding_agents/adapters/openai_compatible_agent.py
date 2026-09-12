"""Coding agent for OpenAI-compatible Chat Completions endpoints.

This adapter deliberately uses the common Chat Completions surface instead of
provider-specific SDKs.  The model returns complete file blocks, which Kapso
writes into the experiment workspace before the evaluator runs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from kapso.execution.coding_agents.base import (
    CodingAgentConfig,
    CodingAgentInterface,
    CodingResult,
)

_FILE_BLOCK = re.compile(
    r"^<<<FILE:[ \t]*(?P<path>[^\r\n>]+?)>>>[ \t]*\r?\n"
    r"(?P<content>.*?)^<<<END FILE>>>[ \t]*(?:\r?\n|$)",
    re.DOTALL | re.MULTILINE,
)


class OpenAICompatibleCodingAgent(CodingAgentInterface):
    """Use an OpenAI-compatible Chat Completions API as a coding agent.

    ``agent_specific`` options:
        base_url: Optional endpoint URL.  Defaults to OPENAI_BASE_URL or
            OPENAI_API_BASE, and then the official OpenAI endpoint.
        api_key_env: Environment variable containing the API key.
        allow_missing_api_key: Use a placeholder key for local endpoints.
        max_output_tokens: Maximum completion length (default 8192).
        temperature: Sampling temperature (default 0.1); None omits it.
        timeout: Request timeout in seconds (default 600).
        read_only: Return text without applying file blocks (inference).
        request_options: Additional Chat Completions request fields.
    """

    def __init__(self, config: CodingAgentConfig):
        super().__init__(config)
        spec = config.agent_specific or {}
        self.workspace: Optional[Path] = None
        self._max_output_tokens = int(spec.get("max_output_tokens", 8192))
        temperature = spec.get("temperature", 0.1)
        self._temperature = None if temperature is None else float(temperature)
        self._timeout = float(spec.get("timeout", 600))
        self._read_only = bool(spec.get("read_only", False))
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
        key_env = spec.get("api_key_env", "OPENAI_API_KEY")
        if not isinstance(key_env, str) or not key_env.strip():
            raise ValueError("api_key_env must name an environment variable")
        self._api_key_env = key_env.strip()
        api_key = os.getenv(self._api_key_env)
        if not api_key:
            if not spec.get("allow_missing_api_key", False):
                raise ValueError(
                    f"{self._api_key_env} not set — openai_compatible needs it"
                )
            api_key = "not-needed"

        base_url = spec.get("base_url")
        if base_url is None:
            base_url = os.getenv(
                spec.get("base_url_env", "OPENAI_BASE_URL")
            ) or os.getenv("OPENAI_API_BASE")
        self._base_url = str(base_url).strip() if base_url else None
        client_options: Dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if self._base_url:
            client_options["base_url"] = self._base_url
        self._client = OpenAI(**client_options)

    def initialize(self, workspace: str) -> None:
        self.workspace = Path(workspace).resolve()

    def generate_code(
        self,
        prompt: str,
        debug_mode: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> CodingResult:
        if self.workspace is None:
            raise RuntimeError(
                "Agent not initialized. Call initialize() first."
            )

        model = self.config.debug_model if debug_mode else self.config.model
        system = self._build_system_prompt()
        options: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
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
        try:
            response = self._client.chat.completions.create(**options)
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) not in (None, "stop"):
                raise RuntimeError(
                    "Incomplete completion: "
                    f"finish_reason={choice.finish_reason}"
                )
            if getattr(choice.message, "tool_calls", None):
                raise RuntimeError(
                    "Tool calls are not supported by this adapter"
                )
            output = choice.message.content or ""
            if not output.strip():
                raise RuntimeError(
                    "OpenAI-compatible endpoint returned empty output"
                )
            files_changed = (
                [] if self._read_only else self._write_files(output)
            )
            usage = getattr(response, "usage", None)
            metadata = {
                "model": model,
                "base_url": self._base_url or "https://api.openai.com/v1",
                "usage": self._usage_dict(usage),
            }
            return CodingResult(
                success=True,
                output=output,
                files_changed=files_changed,
                metadata=metadata,
            )
        except Exception as error:
            return CodingResult(success=False, output="", error=str(error))

    def _build_system_prompt(self) -> str:
        if self._read_only:
            return (
                "Answer the user's request directly in the requested format. "
                "You have no filesystem, shell, or web-search tools. "
                "Do not claim to have read files, run code, "
                "or searched the web."
            )
        return (
            "You are an expert programmer implementing the requested change. "
            "You can only use context supplied in the prompt; you cannot read "
            "the workspace or run commands. "
            "Never invent evaluation results. "
            "Return each changed file in this exact format:\n"
            "<<<FILE: relative/path/to/file>>>\n"
            "complete file contents\n"
            "<<<END FILE>>>\n"
            "Use repository-relative paths, include complete contents, and "
            "return no other file-edit format."
        )

    def _write_files(self, output: str) -> List[str]:
        blocks = list(_FILE_BLOCK.finditer(output))
        if not blocks:
            raise ValueError("No complete <<<FILE: ...>>> blocks in response")
        remainder = _FILE_BLOCK.sub("", output)
        if "<<<FILE:" in remainder or "<<<END FILE>>>" in remainder:
            raise ValueError("Malformed or incomplete file blocks in response")
        # Validate the entire response before touching any workspace files.
        pending = {}
        for block in blocks:
            path = self._safe_path(block.group("path"))
            if path in pending:
                raise ValueError(f"Duplicate file path in response: {path}")
            if path.exists() and not path.is_file():
                raise ValueError(f"File path is not a regular file: {path}")
            if any(
                parent.exists() and not parent.is_dir()
                for parent in path.parents
            ):
                raise ValueError(f"File parent is not a directory: {path}")
            pending[path] = block.group("content")
        if any(
            parent in pending for path in pending for parent in path.parents
        ):
            raise ValueError(
                "Conflicting file and directory paths in response"
            )
        changed = []
        for path, content in pending.items():
            if path.exists() and path.read_text(encoding="utf-8") == content:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            changed.append(str(path))
        return changed

    def _safe_path(self, relative: Optional[str]) -> Path:
        if not relative or self.workspace is None:
            raise ValueError("model response did not provide a file path")
        candidate = Path(relative.strip().strip("`\"'"))
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or ".git" in candidate.parts
        ):
            raise ValueError(
                f"unsafe file path from model response: {relative!r}"
            )
        resolved = (self.workspace / candidate).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError as error:
            raise ValueError(
                f"unsafe file path from model response: {relative!r}"
            ) from error
        if ".git" in resolved.relative_to(self.workspace).parts:
            raise ValueError(
                f"unsafe file path from model response: {relative!r}"
            )
        return resolved

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
