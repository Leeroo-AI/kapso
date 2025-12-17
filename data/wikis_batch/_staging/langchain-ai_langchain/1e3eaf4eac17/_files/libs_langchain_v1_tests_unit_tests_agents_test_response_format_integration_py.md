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

**Purpose:** Integration tests validating response format behavior with real OpenAI models (ChatOpenAI), testing both native structured output and tool-based output strategies.

**Mechanism:** Uses VCR (Video Cassette Recorder) to record and replay HTTP interactions with OpenAI API, eliminating need for live API calls during testing. Tests are parameterized to run with both `use_responses_api=False` (legacy) and `use_responses_api=True` (newer API). Three main test scenarios:
1. `test_inference_to_native_output`: Verifies auto-detection of native structured output support (4 messages)
2. `test_inference_to_tool_output`: Tests explicit ToolStrategy forcing tool-based output (5 messages)
3. `test_strict_mode`: Validates that ProviderStrategy's `strict=True` flag is properly passed to OpenAI's API

Uses mocked API key when `OPENAI_API_KEY` not in environment. Includes detailed instructions for re-recording cassettes and decompressing/reading compressed YAML cassettes.

**Significance:** Ensures the agent framework correctly integrates with external LLM providers, validating that strategy inference works with real models and that API-specific features (like OpenAI's strict mode) are properly supported. Critical for maintaining compatibility across different OpenAI API versions.
