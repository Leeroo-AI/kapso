# File: `libs/langchain_v1/tests/unit_tests/agents/test_response_format.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 875 |
| Classes | `WeatherBaseModel`, `WeatherDataclass`, `WeatherTypedDict`, `LocationResponse`, `LocationTypedDict`, `TestResponseFormatAsModel`, `TestResponseFormatAsToolStrategy`, `TestResponseFormatAsProviderStrategy`, `TestDynamicModelWithResponseFormat`, `CustomModel`, `ModelSwappingMiddleware` |
| Functions | `get_weather`, `get_location`, `test_union_of_types` |
| Imports | collections, dataclasses, json, langchain, langchain_core, pydantic, pytest, tests, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests structured output handling for agents via the `response_format` parameter, covering Pydantic models, dataclasses, TypedDicts, and JSON schemas with both ToolStrategy and ProviderStrategy approaches.

**Mechanism:** Implements comprehensive test suite with three main test classes:
- `TestResponseFormatAsModel`: Tests passing response formats directly (auto-detects strategy)
- `TestResponseFormatAsToolStrategy`: Tests explicit ToolStrategy usage, including union types, error handling with retry, and custom error messages
- `TestResponseFormatAsProviderStrategy`: Tests ProviderStrategy for models that support native structured output, including strict mode flag
- `TestDynamicModelWithResponseFormat`: Tests middleware that can swap models dynamically and ensures strategy resolution is deferred

Uses `FakeToolCallingModel` to simulate model responses with controlled tool calls and structured outputs. Tests validation error handling, multiple structured output errors, and retry mechanisms with configurable error handling (bool, tuple of exceptions, callable, or string).

**Significance:** Critical validation of the agent's structured output capabilities, ensuring proper type coercion, schema validation, error recovery, and strategy selection based on model capabilities. Validates that middleware can modify models before strategy resolution, ensuring flexibility in agent architectures.
