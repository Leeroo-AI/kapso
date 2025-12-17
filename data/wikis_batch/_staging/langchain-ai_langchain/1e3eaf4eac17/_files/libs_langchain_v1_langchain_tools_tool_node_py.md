# File: `libs/langchain_v1/langchain/tools/tool_node.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 20 |
| Imports | langgraph |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides backwards-compatible re-exports of tool node functionality that has been moved to LangGraph, maintaining API compatibility while delegating to the canonical implementation.

**Mechanism:** Re-exports five key components from langgraph.prebuilt and langgraph.prebuilt.tool_node:

1. **From langgraph.prebuilt**:
   - `InjectedState`: Marker for injecting agent state into tools
   - `InjectedStore`: Marker for injecting persistent store into tools
   - `ToolRuntime`: Runtime context container for tool execution

2. **From langgraph.prebuilt.tool_node**:
   - `ToolCallRequest`: Request object for tool invocations
   - `ToolCallWithContext`: Tool call with agent state context
   - `ToolCallWrapper`: Function wrapper for tool call middleware

The actual ToolNode class is imported but not re-exported (marked as _ToolNode with F401 noqa), as it's intended to be used directly from LangGraph.

**Significance:** This module serves as a compatibility layer that:
- Maintains stable import paths for downstream code
- Allows gradual migration to direct LangGraph imports
- Prevents breaking changes for existing users
- Documents the relationship between LangChain and LangGraph components

The re-export strategy is common in evolving codebases where functionality is refactored into separate packages but backwards compatibility must be maintained. This approach allows the tool execution infrastructure to live in LangGraph (where it belongs architecturally) while still being accessible through the langchain.tools namespace.
