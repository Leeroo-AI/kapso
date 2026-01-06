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

**Purpose:** Tests the ContextEditingMiddleware functionality that manages conversation context size by clearing tool outputs and inputs.

**Mechanism:** Uses a custom `_TokenCountingChatModel` to deterministically count tokens, creates test scenarios with various AI and ToolMessages, then verifies that ClearToolUsesEdit correctly: (1) only edits when token trigger threshold is exceeded, (2) clears tool outputs and optionally inputs with placeholder text, (3) respects the "keep" parameter to preserve recent tool results, (4) excludes specified tools from clearing, and (5) maintains original request immutability. Tests cover both sync and async middleware execution paths.

**Significance:** Validates a critical middleware for managing context window limits in long-running agent conversations, ensuring the agent can continue functioning even when tool outputs would exceed model token limits while preserving important recent context.
