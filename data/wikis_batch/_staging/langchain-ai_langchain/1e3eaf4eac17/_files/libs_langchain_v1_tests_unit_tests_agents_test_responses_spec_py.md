# File: `libs/langchain_v1/tests/unit_tests/agents/test_responses_spec.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 148 |
| Classes | `ToolCalls`, `AssertionByInvocation`, `TestCase`, `Employee` |
| Functions | `test_responses_integration_matrix` |
| Imports | __future__, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Spec-driven integration tests for agent responses with structured output formats, testing real OpenAI model behavior against predefined test cases loaded from external spec files (currently skipped as `langgraph.prebuilt.responses` is not available).

**Mechanism:** Would implement matrix-style testing approach if enabled:
- Loads test cases from external spec files using `load_spec("responses", as_model=TestCase)`
- Each test case defines: response format schemas, prompts, expected tool calls, expected messages, and expected structured responses
- Tests HR assistant agent with employee data (Sabine, Henrik, Jessica) and tools for querying roles and departments
- Uses mocked tools to track call counts and validate behavior
- Supports both single schema and union schema (`oneOf`) response formats
- One test case marked as xfail for undefined behavior when model cannot conform to any structured response format

Parameterized test function runs complete interaction scenarios and validates tool call counts, LLM request counts, final message content, and structured response JSON.

**Significance:** Provides high-level behavioral validation of the complete agent pipeline with structured outputs, ensuring that agents correctly use tools and produce structured responses matching expected schemas. Spec-driven approach enables comprehensive testing across multiple scenarios without duplicating test code.
