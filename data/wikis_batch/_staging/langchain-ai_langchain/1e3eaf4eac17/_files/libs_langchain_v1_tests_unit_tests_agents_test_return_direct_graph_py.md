# File: `libs/langchain_v1/tests/unit_tests/agents/test_return_direct_graph.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 73 |
| Functions | `test_agent_graph_without_return_direct_tools`, `test_agent_graph_with_return_direct_tool`, `test_agent_graph_with_mixed_tools` |
| Imports | langchain, langchain_core, model, syrupy |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests that agent graph structure correctly reflects the presence of `return_direct=True` tools by validating Mermaid diagram edges to the end node.

**Mechanism:** Creates three test scenarios using snapshot testing (syrupy):
1. `test_agent_graph_without_return_direct_tools`: Normal tools only, verifies NO edge from tools to `__end__`
2. `test_agent_graph_with_return_direct_tool`: One `return_direct=True` tool, verifies edge TO `__end__` exists
3. `test_agent_graph_with_mixed_tools`: Mix of both tool types, verifies edge TO `__end__` exists

Each test creates an agent with `FakeToolCallingModel` and tools with specified `return_direct` flag, then calls `agent.get_graph().draw_mermaid()` to generate a Mermaid diagram. Snapshot assertions ensure the graph structure matches expected topology.

**Significance:** Validates that the agent graph compiler correctly handles `return_direct=True` tools, which should bypass the agent loop and return results immediately to the caller. The graph structure must reflect this by including a direct edge from the tools node to the end node, enabling the executor to short-circuit execution when these tools are invoked.
