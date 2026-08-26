# CLI inference — every completion is a coding-agent CLI session.
#
# Design: docs/research/cli-only-inference-design.md (user decisions
# 2026-08-25: convert all non-embedding completions; default codex /
# gpt-5.6-sol / xhigh; no API fallback — a missing CLI fails loud).
#
# CliInference is a drop-in replacement for LLMBackend at every seam that
# previously made direct completions: the completion methods keep their
# names and return types but run as read-only coding-agent sessions, while
# embeddings, model-role resolution, and the cost meter delegate to an
# inner LLMBackend (embeddings are explicitly out of scope of the
# conversion). Callers select a role with one `role=` kwarg; the role's
# spec (cli, model, effort, sandbox, web_search, timeout) comes from the
# platform config's top-level `inference:` block — Rule 1: the packaged
# config is the single source, and self-constructing consumers (the
# researcher, KG backends) resolve it via default_inference_config().

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from kapso.core.config import load_config
from kapso.core.llm import LLMBackend
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.factory import CodingAgentFactory

# The packaged platform config carries the inference block (Rule 1).
_PACKAGED_CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")


def default_inference_config() -> Dict[str, Any]:
    """The platform `inference:` block from the packaged config."""
    return load_config(_PACKAGED_CONFIG_PATH)["inference"]


class CliInference:
    """LLMBackend's surface, served by coding-agent CLI sessions.

    Completion methods run one read-only session each and return the
    session's final output text. Empty output or a failed session RAISES
    (Rule 2): a silent empty completion is how the research gate once
    poisoned an entire E2E run. Embeddings / resolve_model /
    get_cumulative_cost delegate to the inner LLMBackend so existing
    consumers (experiment store, budget meter) keep working unchanged.
    """

    def __init__(
        self,
        inference: Optional[Dict[str, Any]] = None,
        models: Optional[Mapping[str, Any]] = None,
        retry_policy: Optional[Any] = None,
        agent_factory=None,
    ):
        self._inference = inference or default_inference_config()
        if "default" not in self._inference:
            raise ValueError("inference config has no 'default' role spec")
        self._backend = LLMBackend(models=models, retry_policy=retry_policy)
        self._agent_factory = agent_factory or CodingAgentFactory
        self._cli_cost = 0.0

    # ------------------------------------------------------------ roles

    def _role_spec(self, role: Optional[str]) -> Dict[str, Any]:
        spec = dict(self._inference["default"])
        if role:
            overrides = (self._inference.get("roles") or {}).get(role)
            if overrides is None:
                raise ValueError(
                    f"unknown inference role {role!r}: expected one of "
                    f"{sorted((self._inference.get('roles') or {}))}"
                )
            spec.update(overrides)
        return spec

    # --------------------------------------------------------- sessions

    def _run_session(
        self, prompt: str, role: Optional[str], web_search: bool = False
    ) -> str:
        spec = self._role_spec(role)
        agent_specific: Dict[str, Any] = {
            "effort": spec["effort"],
            "timeout": spec["timeout_seconds"],
            "streaming": False,
        }
        if spec["cli"] == "codex":
            agent_specific["sandbox"] = spec["sandbox"]
            agent_specific["web_search"] = bool(
                web_search or spec.get("web_search")
            )
        else:
            agent_specific["auth_mode"] = spec.get("auth_mode", "oauth")
        agent = self._agent_factory.create(CodingAgentConfig(
            agent_type=spec["cli"],
            model=spec["model"],
            debug_model=spec["model"],
            agent_specific=agent_specific,
        ))
        scratch = tempfile.mkdtemp(prefix="kapso-inference-")
        agent.initialize(scratch)
        result = agent.generate_code(
            prompt, timeout_seconds=spec["timeout_seconds"]
        )
        self._cli_cost += float(agent.get_cumulative_cost() or 0.0)
        agent.cleanup()
        output = (result.output or "").strip()
        if not result.success or not output:
            raise RuntimeError(
                f"CLI inference session failed (role={role or 'default'}, "
                f"cli={spec['cli']}, model={spec['model']}, "
                f"success={result.success}, output_chars={len(output)}): "
                f"{result.error or 'empty output'}"
            )
        return output

    @staticmethod
    def _flatten(messages: List[Dict[str, str]]) -> str:
        """Messages -> one prompt, system content first (a CLI session has
        no separate system channel). Content rides WHOLE (Rule 6)."""
        ordered = (
            [m for m in messages if m.get("role") == "system"]
            + [m for m in messages if m.get("role") != "system"]
        )
        return "\n\n".join(str(m.get("content", "")) for m in ordered)

    # ---------------------------------------------- completion surface

    def llm_completion(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        role: Optional[str] = None,
        **_ignored: Any,
    ) -> str:
        # `model` and sampling kwargs are legacy surface: the role spec is
        # the single source of the session's model/effort (Rule 1).
        return self._run_session(self._flatten(messages or []), role)

    def llm_completion_with_system_prompt(
        self,
        model: Optional[str] = None,
        system_prompt: str = "",
        user_message: str = "",
        role: Optional[str] = None,
        **_ignored: Any,
    ) -> str:
        return self._run_session(
            self._flatten([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]),
            role,
        )

    def llm_completion_with_web_search(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        search_context_size: str = "medium",
        role: Optional[str] = "research",
        **_ignored: Any,
    ) -> str:
        # Depth travels as prose — the CLI's native web search has no
        # context-size parameter.
        depth_note = (
            "Research depth: verify each load-bearing claim against "
            "primary sources." if search_context_size == "high"
            else "Research depth: survey broadly; cite sources."
        )
        prompt = self._flatten(messages or []) + "\n\n" + depth_note
        return self._run_session(prompt, role, web_search=True)

    # ------------------------------------------------- delegated surface

    def create_embedding(self, text: str, model: Optional[str] = None):
        return self._backend.create_embedding(text, model)

    def resolve_model(self, model: Optional[str], default_role: str = "embedding"):
        return self._backend.resolve_model(model, default_role=default_role)

    def get_cumulative_cost(self) -> float:
        return self._backend.get_cumulative_cost() + self._cli_cost
