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

**Purpose:** Defines the core type system and abstractions for the agent middleware framework, including the AgentMiddleware base class, hook lifecycle methods, state schemas, request/response wrappers, and decorator utilities for creating middleware from functions.

**Mechanism:** This file establishes a comprehensive middleware system through several key components: (1) AgentState TypedDict with messages, jump_to for control flow, and structured_response for output; (2) ModelRequest/ModelResponse dataclasses that encapsulate model invocation details with immutable override() pattern; (3) AgentMiddleware base class defining 8 hook points (before_agent, before_model, after_model, after_agent, wrap_model_call, wrap_tool_call, and async variants) for intercepting agent execution; (4) OmitFromSchema annotations for controlling input/output visibility; (5) Decorator functions (before_model, after_model, etc.) that dynamically create middleware classes from simple functions; (6) hook_config decorator for specifying jump destinations. The system uses Protocol types for type-safe function signatures and supports both sync and async execution.

**Significance:** This is the foundational infrastructure file that enables the entire middleware ecosystem. It defines the contract that all middleware must follow, provides ergonomic decorator-based middleware creation for simple cases while allowing full class-based middleware for complex scenarios, and implements the state management and control flow primitives (jump_to, OmitFromSchema, PrivateStateAttr) that middleware uses to interact with the agent graph. The careful separation of input/output state schemas, the immutable request/response pattern, and the comprehensive hook coverage make this a production-grade extensibility system. Every middleware file in this directory depends on types and patterns defined here, making it the cornerstone of agent customization in LangChain.
