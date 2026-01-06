# File: `libs/langchain_v1/langchain/tools/tool_node.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 20 |
| Imports | langgraph |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backwards compatibility module re-exporting LangGraph's tool execution utilities for use in LangChain.

**Mechanism:** Re-exports InjectedState, InjectedStore, ToolRuntime, and internal tool node classes (ToolCallRequest, ToolCallWithContext, ToolCallWrapper) from langgraph.prebuilt modules. The ToolNode class itself is imported but not exported in __all__.

**Significance:** Integration layer bridging LangChain's tool abstraction with LangGraph's execution model. Enables LangChain tools to access graph state and stores during execution. This file exists primarily for backwards compatibility, allowing existing code to import these utilities from langchain.tools instead of requiring direct langgraph imports.
