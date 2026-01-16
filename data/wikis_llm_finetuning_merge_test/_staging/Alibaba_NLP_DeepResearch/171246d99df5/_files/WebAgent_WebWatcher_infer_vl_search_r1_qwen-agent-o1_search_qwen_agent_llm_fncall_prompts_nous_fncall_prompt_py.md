# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 208 |
| Classes | `NousFnCallPrompt` |
| Functions | `remove_incomplete_special_tokens`, `extract_fn` |
| Imports | copy, json, qwen_agent, typing |

## Understanding

**Status:** Explored

**Purpose:** Implements the Nous/Hermes-style function calling prompt format using XML-like tags for tool calls and responses.

**Mechanism:** The `NousFnCallPrompt` class extends `BaseFnCallPrompt` and implements: (1) `preprocess_fncall_messages()` - converts structured function calls to `<tool_call>\n{json}\n</tool_call>` format where json contains {"name": fn_name, "arguments": args}; tool responses become `<tool_response>\n{result}\n</tool_response>` wrapped in USER messages; injects a system prompt (FN_CALL_TEMPLATE) describing available tools in `<tools></tools>` XML tags with JSON function signatures; (2) `postprocess_fncall_messages()` - parses model output by splitting on `<tool_call>` and `</tool_call>` tags, extracting the JSON function call specification, and reconstructing FunctionCall objects. The `extract_fn()` helper enables partial function name/args extraction for streaming output support.

**Significance:** Provides compatibility with Nous/Hermes-family models and other LLMs trained on similar XML-based tool calling formats. This format is widely adopted in the open-source LLM community, making it important for model interoperability. The clean XML structure makes it easy for models to learn and produce valid tool calls.
