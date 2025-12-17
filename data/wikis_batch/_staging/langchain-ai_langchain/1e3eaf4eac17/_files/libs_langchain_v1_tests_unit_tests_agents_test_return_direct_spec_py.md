# File: `libs/langchain_v1/tests/unit_tests/agents/test_return_direct_spec.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 107 |
| Classes | `TestCase` |
| Functions | `test_return_direct_integration_matrix` |
| Imports | __future__, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Spec-driven integration tests validating `return_direct` tool behavior with structured output formats using real OpenAI models (currently skipped as `langgraph.prebuilt.responses` is not available).

**Mechanism:** Would implement matrix testing if enabled:
- Loads test cases from external spec files using `load_spec("return_direct", as_model=TestCase)`
- Tests polling scenario where agent repeatedly calls `poll_job` tool until status becomes "succeeded"
- Each test case specifies: `return_direct` flag, optional response_format, expected tool call count, expected last message, and expected structured response
- Mock tool returns `{"status": "pending"}` for first 9 calls, then `{"status": "succeeded", "attempts": 10}`
- Uses system prompt to instruct agent to poll until succeeded and report attempt count

Validates that `return_direct=True` tools correctly bypass agent loop and return immediately, while `return_direct=False` tools allow continued iteration. Tests interaction between `return_direct` and structured output formats.

**Significance:** Ensures `return_direct` functionality works correctly in real-world scenarios with polling patterns and structured outputs. Critical for validating that tools can bypass the agent reasoning loop when appropriate, improving efficiency and reducing unnecessary LLM calls.
