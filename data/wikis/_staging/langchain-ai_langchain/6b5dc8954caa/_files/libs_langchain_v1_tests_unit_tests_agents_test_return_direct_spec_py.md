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

**Purpose:** Integration tests for return_direct tool behavior with structured response formats. Currently skipped due to unavailable langgraph.prebuilt.responses dependency.

**Mechanism:** Would use parametrized test cases from a spec file to test various combinations of return_direct flag and response_format settings. Tests would verify tool call counts, last message content, and structured response handling using OpenAI's GPT-4 model with a polling scenario where a tool is called multiple times until success.

**Significance:** Ensures return_direct tools work correctly with structured outputs and response formatting. Important for validating edge cases in agent behavior when tools bypass normal agent processing, but currently disabled pending dependency availability.
