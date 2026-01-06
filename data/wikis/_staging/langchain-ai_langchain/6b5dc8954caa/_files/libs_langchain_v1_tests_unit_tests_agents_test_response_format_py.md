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

**Purpose:** Tests structured output (response_format) functionality in create_agent across all supported schema types and strategies.

**Mechanism:** Comprehensive test suite organized into four main test classes: TestResponseFormatAsModel (raw schemas), TestResponseFormatAsToolStrategy (explicit ToolStrategy), TestResponseFormatAsProviderStrategy (explicit ProviderStrategy), and TestDynamicModelWithResponseFormat (middleware model swapping). Tests cover Pydantic models, dataclasses, TypedDicts, and JSON schemas. Validates proper parsing, error handling with/without retry, custom error messages, union types (oneOf), multiple structured outputs error handling, validation errors, and the strict flag for provider schemas. Uses FakeToolCallingModel to simulate different response patterns.

**Significance:** Critical test coverage for the structured output feature that allows agents to return typed responses. Ensures proper strategy selection (ToolStrategy vs ProviderStrategy), validates error handling and retry mechanisms, and confirms that middleware can modify models before strategy resolution. Essential for type-safe agent outputs and reliable parsing of LLM responses into structured data formats.
