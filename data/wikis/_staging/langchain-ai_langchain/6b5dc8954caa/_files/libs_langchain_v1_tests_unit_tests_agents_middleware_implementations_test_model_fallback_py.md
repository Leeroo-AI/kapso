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

**Purpose:** Tests the ModelFallbackMiddleware that automatically tries fallback models when the primary model fails.

**Mechanism:** Creates custom test models (FailingPrimaryModel, AlwaysFailingModel) that raise exceptions on invocation, then uses wrap_model_call/awrap_model_call to verify: (1) primary model is used when successful, (2) fallback models are tried in sequence when primary fails, (3) appropriate exceptions are raised when all models fail, (4) middleware correctly swaps models using request.override() method, and (5) ModelRequest properly warns on direct attribute assignment (deprecated) while override() method works without warnings. Tests both sync/async execution and integration with create_agent to verify end-to-end behavior.

**Significance:** Validates critical reliability mechanism that enables production agents to automatically recover from model API failures by falling back to alternative models, improving system resilience and availability.
