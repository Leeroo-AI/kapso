# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/constant.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 8 |

## Understanding

**Status:** Explored

**Purpose:** Defines the set of supported API arguments for OpenAI-compatible API calls, serving as a whitelist for valid parameters.

**Mechanism:** Contains a single constant `SUPPORT_ARGS`, a Python set with 24 parameter names including:
- Core parameters: `model`, `messages`, `stream`
- Sampling parameters: `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `seed`, `min_p`, `min_a`
- Token limits: `max_completion_tokens`, `max_tokens`
- Response format: `response_format`, `n`, `logprobs`, `top_logprobs`, `stop`
- Function calling: `tools`, `tool_choice`, `function_call`, `functions`
- Multimodal: `modalities`, `audio`
- Metadata: `user`, `tenant`, `baseUrl`, `logit_bias`

**Significance:** Configuration utility that acts as a validation filter. Used by `OpenAIAPIClient.call()` to filter out unsupported keyword arguments before making API requests, ensuring only valid OpenAI-compatible parameters are sent to the API endpoint.
