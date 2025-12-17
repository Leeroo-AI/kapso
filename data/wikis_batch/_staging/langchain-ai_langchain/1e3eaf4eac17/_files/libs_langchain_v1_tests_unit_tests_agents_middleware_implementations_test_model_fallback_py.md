# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_fallback.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 357 |
| Classes | `FailingPrimaryModel`, `FailingModel`, `AlwaysFailingModel`, `AsyncFailingPrimaryModel`, `AsyncFailingModel`, `AsyncAlwaysFailingModel`, `FailingModel`, `SuccessModel`, `AlwaysFailingModel` |
| Functions | `test_primary_model_succeeds`, `test_fallback_on_primary_failure`, `test_multiple_fallbacks`, `test_all_models_fail`, `test_primary_model_succeeds_async`, `test_fallback_on_primary_failure_async`, `test_multiple_fallbacks_async`, `test_all_models_fail_async`, `... +4 more` |
| Imports | __future__, langchain, langchain_core, langgraph, pytest, tests, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive unit tests for ModelFallbackMiddleware, validating automatic fallback to alternative models when the primary model fails. Tests cover single and multiple fallbacks, both synchronous and asynchronous execution, and integration with agent systems.

**Mechanism:** Uses custom test model classes that simulate various failure scenarios:
- **FailingPrimaryModel/AlwaysFailingModel**: Raise exceptions in `_generate` to simulate model failures
- **AsyncFailingModel**: Async versions for testing async fallback chains
- **GenericFakeChatModel**: Successful models for testing fallback success

Test patterns validate:
- **Success path**: Primary model succeeds, fallbacks not used
- **Single fallback**: Primary fails, fallback succeeds
- **Multiple fallbacks**: Sequential fallback attempts until one succeeds
- **Exhaustion**: All models fail, final exception propagates
- **Async equivalents**: All patterns tested with async model calls
- **Integration tests**: Full agent execution with fallback middleware and checkpointer
- **ModelRequest immutability**: Tests verify that `override()` method is preferred over direct attribute assignment (triggers deprecation warnings)

Mock handlers invoke models and return responses, allowing middleware to intercept and retry with fallbacks.

**Significance:** Critical for production agent reliability, enabling graceful degradation when models experience failures, rate limits, or outages. Supports multiple fallback chains for high availability deployments. The async support ensures fallback logic works correctly in concurrent agent systems. The ModelRequest immutability tests ensure future-proof API design.
