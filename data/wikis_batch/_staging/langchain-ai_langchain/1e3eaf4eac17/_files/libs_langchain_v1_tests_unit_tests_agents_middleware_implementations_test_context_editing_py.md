# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_context_editing.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 451 |
| Classes | `_TokenCountingChatModel` |
| Functions | `test_no_edit_when_below_trigger`, `test_clear_tool_outputs_and_inputs`, `test_respects_keep_last_tool_results`, `test_exclude_tools_prevents_clearing`, `test_no_edit_when_below_trigger_async`, `test_clear_tool_outputs_and_inputs_async`, `test_respects_keep_last_tool_results_async`, `test_exclude_tools_prevents_clearing_async` |
| Imports | __future__, collections, langchain, langchain_core, langgraph, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive unit tests for the ContextEditingMiddleware, verifying token limit management and tool output clearing for agent conversations. Tests cover token counting triggers, selective tool output clearing, and preservation of recent tool results.

**Mechanism:** Utilizes a custom `_TokenCountingChatModel` that deterministically counts tokens based on message content length. Tests create mock agent states with tool calls and tool messages, then verify that the middleware correctly applies edits when token counts exceed configured triggers. Covers both synchronous and asynchronous execution paths with tests for:
- Token count triggering (no edits when below threshold)
- Clearing tool outputs and inputs with placeholders
- Respecting `keep` parameter to preserve recent tool results
- Excluding specific tools from clearing via `exclude_tools`
- Custom placeholder text and token count methods

**Significance:** Critical for ensuring agents can handle long conversations by intelligently managing context size. The middleware prevents context overflow by selectively clearing older tool outputs while preserving recent relevant information, essential for production agent systems with token limits.
