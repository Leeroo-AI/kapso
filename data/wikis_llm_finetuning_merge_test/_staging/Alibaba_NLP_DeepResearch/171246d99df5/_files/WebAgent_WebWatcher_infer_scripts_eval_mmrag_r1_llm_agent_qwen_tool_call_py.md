# File: `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/llm_agent/qwen_tool_call.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 60 |
| Classes | `Qwen_agent` |
| Imports | copy, json, os, qwen_agent, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a customized Qwen agent wrapper that exposes tool-calling capabilities for web search, image search, URL visiting, and code interpretation without requiring a full LLM backend.

**Mechanism:** The `Qwen_agent` class extends qwen_agent's `Agent` base class. Key features: (1) Disables several CSI/IDP features via environment variables at import time. (2) Overrides `_run()` to return nothing (no LLM generation needed). (3) Overrides `_call_tool()` to handle file access for tools that need it, extracting files from messages when required. (4) Supports tools: web_search, VLSearchImage, visit, code_interpreter, PythonInterpreter, google_scholar, google_search. (5) The `__main__` block provides test examples for tool invocation.

**Significance:** Adapter layer that bridges the LLMGenerationManager with Qwen-agent's tool ecosystem. Allows the generation manager to call tools independently without full agent execution, enabling fine-grained control over when and how tools are invoked during the multi-turn generation loop.
