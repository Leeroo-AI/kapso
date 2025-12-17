# File: `libs/langchain_v1/langchain/agents/middleware/tool_selection.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 320 |
| Classes | `_SelectionRequest`, `LLMToolSelectorMiddleware`, `ToolSelectionResponse` |
| Imports | __future__, dataclasses, langchain, langchain_core, logging, pydantic, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Uses an LLM to intelligently filter available tools down to only the most relevant ones for the user's query before calling the main model, reducing token usage and improving focus when agents have access to many tools (e.g., 50+ tools where only 3-5 are relevant to current task).

**Mechanism:** The LLMToolSelectorMiddleware intercepts model calls in `wrap_model_call`, extracts the last user message, creates a dynamic Pydantic schema with Literal types for each available tool name (annotated with descriptions), invokes a selection model (defaults to agent's model) with structured output to choose relevant tools, validates the selections against available tools, and modifies the ModelRequest to include only selected tools plus any always-included tools before passing to the handler. The selection uses a Union of Annotated Literals for type-safe tool names, and the max_tools parameter limits selections to top N most relevant.

**Significance:** This middleware addresses a critical scalability challenge: as agents gain access to more tools, the context length and token costs grow linearly, and models may struggle to focus on the right tools. By delegating tool selection to a (potentially smaller/cheaper) model, it dramatically reduces the main model's tool list, improving both performance and cost. The always_include parameter ensures critical tools (e.g., task_complete) are never filtered out. The structured output approach with Literal unions provides strong type safety and clear error messages when models select invalid tools. This is particularly valuable for specialized agents with domain-specific tool suites where most tools are irrelevant to any given query.
