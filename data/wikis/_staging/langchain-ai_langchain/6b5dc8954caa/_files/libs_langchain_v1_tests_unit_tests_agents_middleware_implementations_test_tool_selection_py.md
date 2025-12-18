# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_selection.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 596 |
| Classes | `FakeModel`, `TestLLMToolSelectorBasic`, `TestMaxToolsLimiting`, `TestAlwaysInclude`, `TestDuplicateAndInvalidTools`, `TestEdgeCases` |
| Functions | `get_weather`, `search_web`, `calculate`, `send_email`, `get_stock_price` |
| Imports | itertools, langchain, langchain_core, pydantic, pytest, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the LLMToolSelectorMiddleware that dynamically filters available tools to a relevant subset before each model call, reducing context size and improving focus.

**Mechanism:** Validates LLM-based tool selection via ToolSelectionResponse schema, max_tools limiting that truncates to first N selected tools, always_include tools that bypass limits and selection, duplicate tool deduplication, empty tool list handling, and both sync/async execution paths. Uses FakeModel to simulate tool selection responses and trace_model_requests wrapper to verify filtered tool lists passed to main model.

**Significance:** Performance optimization test suite enabling agents to work with large tool inventories by intelligently selecting relevant tools per task, reducing token usage and improving model focus without manual curation.
