# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_diagram.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 192 |
| Classes | `NoopOne`, `NoopTwo`, `NoopThree`, `NoopFour`, `NoopFive`, `NoopSix`, `NoopSeven`, `NoopEight`, `NoopNine`, `NoopTen`, `NoopEleven` |
| Functions | `test_create_agent_diagram` |
| Imports | collections, langchain, langchain_core, syrupy, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Snapshot testing for agent graph diagrams with varying middleware configurations

**Mechanism:** Creates agents with different middleware combinations and validates their Mermaid graph representations using snapshot assertions. Tests progressively add middleware:

**Middleware Patterns:**
- Zero middleware (baseline agent)
- 1-3 before_model hooks (NoopOne, NoopTwo, NoopThree)
- 1-3 after_model hooks (NoopFour, NoopFive, NoopSix)
- Combined before+after hooks (NoopSeven, NoopEight, NoopNine)
- Middleware with wrap_model_call (NoopTen, NoopEleven)

Each configuration generates a Mermaid diagram via `get_graph().draw_mermaid()` and compares against stored snapshots. The test systematically covers:
- Single hook types (before-only, after-only)
- Multiple hooks of the same type
- Mixed hook types
- Hooks with wrap_model_call middleware

**Significance:** Critical for visual regression testing of graph structure - ensures that middleware additions create the expected nodes and edges in the agent execution graph. Changes to graph topology are caught by snapshot mismatches, preventing unintended alterations to agent flow diagrams that users rely on for understanding agent behavior.
