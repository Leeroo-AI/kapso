# File: `libs/langchain_v1/langchain/tools/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 22 |
| Imports | langchain, langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Public entrypoint for tool abstractions, exposing base classes, decorators, and runtime utilities for building and executing tools.

**Mechanism:** Re-exports core tool primitives from langchain_core (BaseTool, tool decorator, ToolException, InjectedToolArg, InjectedToolCallId) and LangGraph-specific runtime utilities from tool_node module (InjectedState, InjectedStore, ToolRuntime).

**Significance:** Primary API surface for working with tools in LangChain. Tools enable LLMs to interact with external systems and APIs. The injected parameters support dependency injection patterns for passing context (state, stores, tool call IDs) to tool functions. ToolRuntime bridges LangChain tools with LangGraph's execution model.
