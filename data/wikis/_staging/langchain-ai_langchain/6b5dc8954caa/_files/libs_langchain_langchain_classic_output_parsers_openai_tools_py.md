# File: `libs/langchain/langchain_classic/output_parsers/openai_tools.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 7 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-exports OpenAI tools API parsers from langchain_core for backward compatibility.

**Mechanism:** Imports and re-exports three OpenAI tools parsers from langchain_core: JsonOutputKeyToolsParser, JsonOutputToolsParser, and PydanticToolsParser.

**Significance:** Compatibility shim for OpenAI's newer tools API (successor to function calling). Maintains stable import paths for code using OpenAI's structured output capabilities.
