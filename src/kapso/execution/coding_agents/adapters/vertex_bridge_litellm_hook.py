"""Loaded by the Vertex bridge's LiteLLM proxy, never by Kapso itself.

LiteLLM routes Vertex AI's open models (zai-org, deepseek, qwen, ...) through
its Llama partner config, whose parameter whitelist lacks ``reasoning_effort``,
so with ``drop_params`` the proxy silently drops the effort the adapter sets.
Importing this module adds the parameter to that whitelist; ``hook`` is the
no-op callback instance that ``litellm_settings.callbacks`` needs to load it.
"""

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import (
    VertexAILlama3Config,
)

_supported_params = VertexAILlama3Config.get_supported_openai_params


def _with_reasoning_effort(self, model):
    params = _supported_params(self, model)
    if "reasoning_effort" not in params:
        params.append("reasoning_effort")
    return params


VertexAILlama3Config.get_supported_openai_params = _with_reasoning_effort
hook = CustomLogger()
