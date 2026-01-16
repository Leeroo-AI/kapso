# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/fncall_prompts/qwen_fncall_prompt.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 389 |
| Classes | `QwenFnCallPrompt` |
| Functions | `get_function_description`, `remove_incomplete_special_tokens`, `remove_trailing_comment_of_fn_args` |
| Imports | copy, json, qwen_agent, typing |

## Understanding

**Status:** Explored

**Purpose:** Implements the native Qwen model function calling format using special Unicode tokens for structured tool invocation.

**Mechanism:** The `QwenFnCallPrompt` class implements the most feature-rich prompt format: (1) Uses distinctive Unicode markers: FN_NAME='FUNCTION', FN_ARGS='ARGS', FN_RESULT='RESULT', FN_EXIT='RETURN'; (2) `preprocess_fncall_messages()` converts function calls to "FUNCTION: name\nARGS: args" format, handles function results with RESULT/RETURN markers, injects bilingual (zh/en) system prompts from FN_CALL_TEMPLATE dict with parallel call variants, supports forced function_choice by prefixing the function name; (3) `postprocess_fncall_messages()` parses output by splitting on "FUNCTION:" markers, extracts function names and arguments, handles multiple parallel function calls, and supports function_choice continuation; (4) `get_function_description()` generates rich tool descriptions with human/model names, descriptions, and parameters in JSON; (5) Helper functions handle streaming edge cases and trailing comments in arguments.

**Significance:** The primary and most sophisticated function calling format designed specifically for Qwen models. Supports parallel function calls, bilingual prompts, function_choice forcing, and code interpreter special formatting. This is the default format for Qwen-based agents in the DeepResearch system, providing the tightest integration with Alibaba's model family while offering the most complete feature set for complex multi-tool research workflows.
