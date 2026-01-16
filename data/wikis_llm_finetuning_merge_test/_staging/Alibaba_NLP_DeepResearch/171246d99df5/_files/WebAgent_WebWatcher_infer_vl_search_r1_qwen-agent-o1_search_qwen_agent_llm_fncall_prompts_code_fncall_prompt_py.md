# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/fncall_prompts/code_fncall_prompt.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 191 |
| Classes | `CodeFnCallPrompt` |
| Functions | `remove_incomplete_special_tokens`, `remove_incomplete_special_tokens_for_fn`, `extract_fn` |
| Imports | copy, json, os, qwen_agent, typing |

## Understanding

**Status:** Explored

**Purpose:** Implements a function calling prompt format specifically designed for code interpreter tools, using Python code blocks as the function call syntax.

**Mechanism:** The `CodeFnCallPrompt` class extends `BaseFnCallPrompt` and implements: (1) `preprocess_fncall_messages()` - converts structured function calls to plaintext using markdown code blocks (```python\n{code}\n```) for function calls and ```output for results; only supports single function (code_interpreter) with no parallel calls; (2) `postprocess_fncall_messages()` - parses model output by detecting FN_START (```python\n) and FN_END (\n```\n) markers, extracting code blocks, and converting them back to FunctionCall objects with name='code_interpreter_http'. Helper functions handle streaming output edge cases by removing incomplete special tokens. The DEFAULT_FN_NAME is 'code_interpreter_http' and FN_STOP_WORDS controls whether to stop at observation output.

**Significance:** Essential component for enabling LLM-based code execution workflows. This format is natural for code-generation models and aligns with how developers typically write code in markdown. Used when the agent needs to execute Python code through a code interpreter tool, making it a key enabler for computational research tasks in the DeepResearch system.
