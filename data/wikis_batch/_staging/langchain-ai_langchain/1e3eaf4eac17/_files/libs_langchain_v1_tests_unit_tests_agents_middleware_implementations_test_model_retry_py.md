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

**Purpose:** Extensive unit tests for ModelRetryMiddleware, validating automatic retry logic with exponential backoff for transient model failures. Tests cover retry configuration, backoff strategies, exception filtering, failure handling, and timing validation.

**Mechanism:** Uses specialized test model classes to simulate various failure patterns:
- **TemporaryFailureModel**: Fails N times before succeeding, tracks attempt counts
- **AlwaysFailingModel**: Always raises exceptions for testing retry exhaustion
- **CustomErrorModel**: Raises custom exceptions with retry-specific attributes

Test coverage organized by functionality:
- **Initialization**: Validates default values, custom configuration, and parameter validation (negative values rejected)
- **Retry behavior**: Tests successful retries after temporary failures, retry exhaustion handling, and zero-retry configurations
- **Backoff strategies**: Validates exponential backoff, constant backoff (factor=0), max delay caps, and jitter variation
- **Exception filtering**: Tests retry-on tuple (specific exception types), custom callable filters (inspect exception attributes)
- **Failure handling**: Tests `on_failure` behaviors (continue with message, error with exception, custom formatter)
- **Timing validation**: Uses `time.time()` to verify actual backoff delays are applied correctly
- **Async support**: Parallel async test suite validates retry logic in async agent execution
- **Middleware composition**: Tests interaction with other middleware in agent pipelines

**Significance:** Essential for production agent resilience against transient API failures, rate limits, and network issues. The configurable retry strategy (max retries, backoff, jitter) allows tuning for different deployment environments and model providers. Exception filtering enables retry only for recoverable errors while failing fast on permanent errors. The comprehensive timing and async tests ensure retry behavior is correct under production load conditions.
