# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/fncall_prompts/nous_fncall_prompt_think.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 283 |
| Classes | `NousFnCallPromptThink` |
| Functions | `remove_incomplete_special_tokens`, `extract_fn` |
| Imports | copy, json, os, qwen_agent, re, typing |

## Understanding

**Status:** Explored

**Purpose:** Extends the Nous function calling format to support chain-of-thought reasoning with `<think></think>` tags, enabling models to reason before making tool calls.

**Mechanism:** The `NousFnCallPromptThink` class builds on `NousFnCallPrompt` with key additions: (1) THINKING_MODE flag enables `<think>` tag processing - during preprocessing, think blocks are stripped from training data; during postprocessing, think content is preserved separately from tool calls; (2) SPECIAL_CODE_MODE (env-configurable) provides special handling for code_interpreter tools - code is extracted from arguments and wrapped in `<code></code>` tags within the tool_call block; (3) Two FN_CALL_TEMPLATEs: standard one instructs model to "think step by step" before tool calls, while FN_CALL_TEMPLATE_WITH_CI adds specific code parameter formatting instructions; (4) `postprocess_fncall_messages()` handles the `</think>` split to separate reasoning from action, and parses `<code>` blocks back into function arguments.

**Significance:** Critical component for implementing reasoning-augmented tool use, aligning with the "o1-style" chain-of-thought approach referenced in the directory name. This enables more deliberate, explainable agent behavior where the model's reasoning process is captured and can be inspected. The code interpreter special handling optimizes code-heavy research workflows by keeping code readable rather than JSON-escaped.
