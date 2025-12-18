# File: `libs/langchain_v1/langchain/agents/middleware/types.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1848 |
| Classes | `_ModelRequestOverrides`, `ModelRequest`, `ModelResponse`, `OmitFromSchema`, `AgentState`, `_InputAgentState`, `_OutputAgentState`, `AgentMiddleware`, `_CallableWithStateAndRuntime`, `_CallableReturningSystemMessage`, `_CallableReturningModelResponse`, `_CallableReturningToolResponse` |
| Functions | `hook_config`, `before_model`, `before_model`, `before_model`, `after_model`, `after_model`, `after_model`, `before_agent`, `... +14 more` |
| Imports | __future__, collections, dataclasses, inspect, langchain_core, langgraph, typing, typing_extensions, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines core type system and base classes for agent middleware architecture.

**Mechanism:** Provides ModelRequest/ModelResponse dataclasses representing model invocation requests/responses. Defines AgentMiddleware base class with lifecycle hooks: before_agent, before_model, after_model, after_agent, and interceptors: wrap_model_call, wrap_tool_call. Implements AgentState TypedDict with messages (add_messages reducer), jump_to (ephemeral control flow), and structured_response fields. Exports decorator functions (before_model, after_model, dynamic_prompt, wrap_model_call, wrap_tool_call) that dynamically create middleware classes from functions. Uses OmitFromSchema annotations to mark private/internal state attributes. Includes Protocol definitions for type-safe middleware composition.

**Significance:** Foundational types file that establishes the middleware abstraction layer for LangChain agents. Defines the contract between agent framework and middleware implementations. Critical for enabling composable, reusable agent behaviors through standardized lifecycle hooks and state management. Provides both class-based and functional programming interfaces via decorators for maximum flexibility.
