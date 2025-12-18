# File: `libs/langchain/langchain_classic/output_parsers/openai_functions.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 13 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-exports OpenAI function calling parsers from langchain_core for backward compatibility.

**Mechanism:** Imports and re-exports four OpenAI function parsers from langchain_core: JsonKeyOutputFunctionsParser, JsonOutputFunctionsParser, PydanticAttrOutputFunctionsParser, and PydanticOutputFunctionsParser.

**Significance:** Compatibility shim maintaining classic import paths for OpenAI function calling support. Critical for existing code using OpenAI's function calling API with LangChain.
