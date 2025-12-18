# File: `libs/langchain_v1/tests/unit_tests/agents/test_return_direct_graph.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 73 |
| Functions | `test_agent_graph_without_return_direct_tools`, `test_agent_graph_with_return_direct_tool`, `test_agent_graph_with_mixed_tools` |
| Imports | langchain, langchain_core, model, syrupy |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests graph structure validation for return_direct tool behavior in agents. Validates that agent graphs correctly include/exclude edges from tools to end nodes based on whether tools have return_direct=True.

**Mechanism:** Uses snapshot testing (syrupy) to compare mermaid diagram representations of agent graphs. Creates agents with different tool configurations (no return_direct tools, all return_direct tools, mixed tools) and validates the resulting graph structure matches expected patterns. Tests rely on FakeToolCallingModel for deterministic behavior.

**Significance:** Critical for ensuring correct graph topology in agent execution. The return_direct flag affects control flow by allowing tools to bypass the agent and return results directly. Graph structure correctness is essential for proper agent behavior and debugging.
