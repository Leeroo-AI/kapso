# File: `libs/langchain_v1/langchain/agents/middleware/tool_selection.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 320 |
| Classes | `_SelectionRequest`, `LLMToolSelectorMiddleware`, `ToolSelectionResponse` |
| Imports | __future__, dataclasses, langchain, langchain_core, logging, pydantic, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Uses an LLM to filter available tools down to most relevant ones before main model call.

**Mechanism:** Implements LLMToolSelectorMiddleware that hooks into wrap_model_call. Dynamically creates Pydantic schema with Union of Literal types (one per tool name) with Field descriptions from tool.description. Invokes selection model (defaults to agent's model) with structured output to choose relevant tools. Processes response to filter original tools list, respecting max_tools limit and always_include list. Returns modified ModelRequest with filtered tools before calling handler for main model invocation.

**Significance:** Optimization middleware that reduces token usage and improves model focus when agents have many tools. Particularly valuable for large tool libraries (20+ tools) where including all tool schemas exceeds context limits or overwhelms the model. Uses smaller/cheaper model for selection to reduce costs while maintaining quality of tool choice in main model.
