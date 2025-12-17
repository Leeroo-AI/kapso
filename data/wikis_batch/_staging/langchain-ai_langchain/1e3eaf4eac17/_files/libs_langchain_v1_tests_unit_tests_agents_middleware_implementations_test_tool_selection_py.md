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

**Purpose:** Tests LLMToolSelectorMiddleware which dynamically filters agent tools using an LLM to select relevant subset based on user query. Tests validate basic selection (sync/async), max_tools limiting, always_include tools (not counted against max_tools), duplicate deduplication, and edge cases. Enables agents with large tool libraries to present only relevant tools to the main agent model, improving reasoning efficiency and reducing context size.

**Mechanism:** Uses FakeModel for both selector model (returns ToolSelectionResponse with selected tool names) and main agent model (uses filtered tools). Tests use trace_model_requests middleware to capture model requests and verify tools lists match expectations. Validates max_tools by having selector return more tools than limit and checking only first N are passed to main model. Tests always_include by verifying those tools appear even when not selected, and don't count against max_tools. Validates deduplication by having selector return duplicate tool names and checking final list has no duplicates.

**Significance:** Solves the "tool explosion" problem for agents with many tools (10s-100s). Large tool lists increase context size, slow inference, and degrade reasoning quality. Dynamic selection provides relevant tool subset (e.g., only weather tools for weather queries), improving model performance. Two-model architecture (selector + main agent) enables specialization: selector optimizes for relevance, main agent focuses on task execution. always_include supports critical tools that should always be available (e.g., fallback tools, control flow tools). Essential for production agents with extensive tooling where presenting all tools every turn is impractical.
