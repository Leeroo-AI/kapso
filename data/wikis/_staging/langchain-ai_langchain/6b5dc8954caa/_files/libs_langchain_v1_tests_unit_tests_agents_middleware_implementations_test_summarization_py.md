# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_summarization.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 889 |
| Classes | `MockChatModel`, `ProfileChatModel`, `ErrorModel`, `ImportErrorProfileModel`, `MockModel`, `NoProfileModel`, `InvalidProfileModel`, `MissingTokensModel`, `InvalidTokenTypeModel`, `ErrorAsyncModel`, `MockAnthropicModel` |
| Functions | `test_summarization_middleware_initialization`, `test_summarization_middleware_no_summarization_cases`, `test_summarization_middleware_helper_methods`, `test_summarization_middleware_summary_creation`, `test_summarization_middleware_trim_limit_none_keeps_all_messages`, `test_summarization_middleware_profile_inference_triggers_summary`, `test_summarization_middleware_token_retention_advances_past_tool_messages`, `test_summarization_middleware_missing_profile`, `... +18 more` |
| Imports | langchain, langchain_core, langgraph, pytest, tests, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the SummarizationMiddleware that automatically condenses conversation history when context limits are approached, preventing token overflow.

**Mechanism:** Validates trigger conditions (tokens/messages/fraction of profile limits), cutoff calculations using binary search for token budgets, safe cutoff points that preserve tool call sequences, summary generation via LLM, message replacement with RemoveMessage markers, profile limit inference from model metadata, and both sync/async execution paths. Tests edge cases like tool message handling, aggressive summarization, and deprecation warnings.

**Significance:** Essential test suite for memory management middleware that enables long-running conversations by intelligently compressing history while maintaining coherence and tool execution context.
