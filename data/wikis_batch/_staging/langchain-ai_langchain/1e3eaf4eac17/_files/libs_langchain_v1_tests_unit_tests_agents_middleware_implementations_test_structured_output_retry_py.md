# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_structured_output_retry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 369 |
| Classes | `StructuredOutputRetryMiddleware`, `WeatherReport` |
| Functions | `get_weather`, `test_structured_output_retry_first_attempt_invalid`, `test_structured_output_retry_exceeds_max_retries`, `test_structured_output_retry_succeeds_first_attempt`, `test_structured_output_retry_validation_error`, `test_structured_output_retry_zero_retries`, `test_structured_output_retry_preserves_messages` |
| Imports | collections, langchain, langchain_core, langgraph, pydantic, pytest, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests a custom middleware implementation that demonstrates retry logic for structured output parsing errors. Shows how to wrap model calls to catch StructuredOutputError exceptions and retry with error feedback. Tests validate retry counts, max retry limits, error message preservation, and success after retries. Serves as an example pattern for implementing custom retry middleware.

**Mechanism:** Implements StructuredOutputRetryMiddleware with wrap_model_call hook that catches StructuredOutputError and appends error feedback to messages before retrying. Uses FakeToolCallingModel with pre-programmed tool call sequences (invalid args, then valid) to simulate structured output validation failures. Tests verify the middleware correctly: retries up to max_retries, includes both AI message and error in feedback, raises error after exhausting retries, and preserves error messages in chat history for model learning.

**Significance:** Demonstrates the agent middleware pattern for handling transient model failures. Structured output parsing is prone to validation errors (wrong types, missing fields), and automatic retry with feedback helps models self-correct. This test file serves as both validation and documentation for implementing custom retry middleware. The pattern of catching specific exceptions, appending feedback, and re-invoking the handler is reusable for other error recovery scenarios (rate limits, API errors, etc.).
