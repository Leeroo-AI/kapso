# Core Module - Shared fundamentals
#
# Contains configuration utilities and LLM backend. Nothing here may import
# from kapso.execution: the gate registry (gated_mcp/presets.py) reads the
# packaged config through kapso.core.config, and the execution package
# imports the gate registry — an execution import here closes that cycle
# on every cold start of the MCP server subprocess.

from kapso.core.llm import (
    DEFAULT_MODEL_ROUTES,
    MODEL_ROLES,
    LLMBackend,
    LLMRetryError,
    ModelRouter,
    RetryPolicy,
    is_transient_llm_error,
)
from kapso.core.config import load_config, load_mode_config

__all__ = [
    # LLM
    "DEFAULT_MODEL_ROUTES",
    "MODEL_ROLES",
    "LLMBackend",
    "LLMRetryError",
    "ModelRouter",
    "RetryPolicy",
    "is_transient_llm_error",
    # Config
    "load_config",
    "load_mode_config",
]
