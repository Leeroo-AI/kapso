# File: `libs/langchain_v1/langchain/agents/factory.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1682 |
| Functions | `create_agent` |
| Imports | __future__, itertools, langchain, langchain_core, langgraph, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides the `create_agent` factory function that builds LLM agents with middleware support, tool calling, and structured output capabilities.

**Mechanism:** Constructs a LangGraph StateGraph with model and tool nodes, chains middleware hooks (before_agent, before_model, after_model, after_agent) into execution pipelines, handles structured output via either ToolStrategy (tool calling) or ProviderStrategy (native JSON mode), and compiles the graph with appropriate edges for agent loop control. Key patterns include middleware composition using decorator-style chaining, auto-detection of model capabilities for structured output, and dynamic tool binding based on middleware configuration.

**Significance:** This is the core implementation of the agent system in langchain v1. It orchestrates all agent components including the LLM, tools, middleware, state management, and structured output handling. The 1682-line implementation handles complex concerns like middleware composition, error handling, async/sync dual support, and conditional graph routing based on tool calls and structured responses.
