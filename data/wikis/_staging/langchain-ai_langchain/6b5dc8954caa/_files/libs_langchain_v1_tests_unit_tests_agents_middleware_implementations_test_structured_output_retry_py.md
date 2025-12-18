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

**Purpose:** Tests middleware that automatically retries model calls when structured output parsing fails, implementing a retry loop with error feedback.

**Mechanism:** Defines StructuredOutputRetryMiddleware that wraps model calls and catches StructuredOutputError exceptions. On failure, appends error feedback to request messages and retries up to max_retries times. Tests verify retry behavior with invalid schema arguments, validation errors, retry exhaustion, and message preservation across attempts.

**Significance:** Example implementation demonstrating how to build resilient structured output workflows by automatically recovering from parsing failures, critical for ensuring agents produce valid structured responses.
