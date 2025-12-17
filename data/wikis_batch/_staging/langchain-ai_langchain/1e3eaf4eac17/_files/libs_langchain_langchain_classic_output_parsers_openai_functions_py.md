# File: `libs/langchain/langchain_classic/output_parsers/openai_functions.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 13 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-export OpenAI function calling output parsers from langchain_core for parsing structured function call responses.

**Mechanism:** Imports and exposes JsonKeyOutputFunctionsParser, JsonOutputFunctionsParser, PydanticAttrOutputFunctionsParser, and PydanticOutputFunctionsParser from langchain_core.output_parsers.openai_functions, providing access to OpenAI-specific function calling parsers.

**Significance:** Provides a convenient import location for parsers that handle OpenAI's function calling feature, which allows LLMs to return structured function calls that can be parsed into JSON or Pydantic objects for programmatic execution.
