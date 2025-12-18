# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_retry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 690 |
| Classes | `TemporaryFailureModel`, `AlwaysFailingModel`, `CustomError`, `CustomErrorModel` |
| Functions | `test_model_retry_initialization_defaults`, `test_model_retry_initialization_custom`, `test_model_retry_invalid_max_retries`, `test_model_retry_invalid_initial_delay`, `test_model_retry_invalid_max_delay`, `test_model_retry_invalid_backoff_factor`, `test_model_retry_working_model_no_retry_needed`, `test_model_retry_failing_model_returns_message`, `... +15 more` |
| Imports | langchain, langchain_core, langgraph, pydantic, pytest, tests, time, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the ModelRetryMiddleware that automatically retries model calls on transient failures with exponential backoff.

**Mechanism:** Uses custom test models (TemporaryFailureModel that fails N times then succeeds, AlwaysFailingModel, CustomErrorModel) to verify: (1) initialization with default/custom parameters and validation, (2) no retries for working models, (3) configurable retry behavior with max_retries, (4) on_failure handling ("continue" returns error message, "error" re-raises, or custom formatter), (5) exponential backoff timing with configurable factor/initial_delay/max_delay/jitter, (6) retry_on filtering (tuple of exception types or callable predicate), (7) proper backoff calculations including max_delay cap and jitter variation, and (8) both sync/async execution. Tests use time measurements to verify actual delays.

**Significance:** Validates essential reliability middleware for handling transient API failures, rate limits, and network issues in production agents with sophisticated retry strategies including exponential backoff and selective retry logic based on exception types.
