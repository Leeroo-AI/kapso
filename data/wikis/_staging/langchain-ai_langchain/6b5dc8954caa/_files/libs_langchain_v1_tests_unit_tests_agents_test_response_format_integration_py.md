# File: `libs/langchain_v1/tests/unit_tests/agents/test_response_format_integration.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 193 |
| Classes | `WeatherBaseModel` |
| Functions | `get_weather`, `test_inference_to_native_output`, `test_inference_to_tool_output`, `test_strict_mode` |
| Imports | langchain, langchain_core, os, pydantic, pytest, typing, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration tests for response_format with real OpenAI models using VCR cassettes for reproducibility.

**Mechanism:** Uses pytest-vcr to record and replay HTTP interactions with OpenAI API. Tests three scenarios with parametrization for use_responses_api flag: (1) inference to native output - verifies ProviderStrategy is auto-selected when model supports it, (2) inference to tool output - verifies ToolStrategy works when explicitly specified, (3) strict mode - verifies the strict flag is properly passed through to the API. Tests use ChatOpenAI with gpt-5 model and validate structured responses are correctly parsed. Includes detailed instructions for re-recording cassettes and inspecting compressed YAML recordings.

**Significance:** Validates that response_format works correctly with actual OpenAI models, not just test doubles. Ensures proper integration between LangChain's response_format abstraction and OpenAI's structured output features. The VCR cassettes make these integration tests fast and reproducible without requiring live API keys during CI/CD, while still validating real-world behavior.
