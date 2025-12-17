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

**Purpose:** Comprehensive tests for SummarizationMiddleware which manages conversation context by summarizing old messages when token/message limits are exceeded. Tests validate trigger conditions (tokens, messages, fractions), token-based retention with binary search, model profile inference, safe cutoff points (avoiding ToolMessage boundaries), message trimming, summary generation (sync/async), error handling, and deprecated parameter warnings.

**Mechanism:** Uses mock chat models (MockChatModel, ProfileChatModel) to simulate summarization and token counting. Tests trigger conditions by mocking token_counter to return values above/below thresholds. Validates binary search algorithm for finding token-based cutoffs by creating messages with known token counts. Tests safe cutoff logic by creating message sequences with ToolMessages that must not be split from their corresponding AIMessages. Verifies profile inference by checking max_input_tokens from model profiles and validating fractional trigger/keep calculations.

**Significance:** Essential for long-running agent conversations that would otherwise exceed context windows. The middleware automatically manages context by summarizing old messages while preserving recent ones, enabling agents to maintain coherent state across extended interactions. Binary search optimization and safe cutoff logic (respecting tool call boundaries) prevent context corruption. Support for both absolute (tokens/messages) and fractional (percentage of max) limits provides flexibility across different model types. Tests ensure robust handling of edge cases (empty messages, profile errors, aggressive summarization) critical for production reliability.
