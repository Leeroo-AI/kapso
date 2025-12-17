# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_composition.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 275 |
| Classes | `TestChainModelCallHandlers` |
| Functions | `create_test_request`, `create_mock_base_handler` |
| Imports | langchain, langchain_core, langgraph, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Test handler composition and chaining behavior for model call middleware

**Mechanism:** Validates the `_chain_model_call_handlers` function that composes multiple middleware handlers into a single execution chain. Tests cover:
- Empty handler lists returning None
- Single and multiple handler compositions (2-3 handlers)
- Execution order verification (outer-before, inner-before, inner-after, outer-after pattern)
- Retry logic with multiple handler invocations
- Error-to-success conversion by outer handlers
- Request modification through the chain
- State and runtime preservation across handlers
- Response normalization to ModelResponse format

Helper functions provide test infrastructure:
- `create_test_request`: Creates ModelRequest instances with default values
- `create_mock_base_handler`: Returns a mock handler that produces ModelResponse

**Significance:** Critical for ensuring middleware composition works correctly - handlers must execute in the proper order (first-to-last on entry, last-to-first on exit), preserve state, and handle retries/errors appropriately. This is the foundation for the middleware system's composability.
