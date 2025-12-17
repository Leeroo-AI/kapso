# File: `libs/langchain/langchain_classic/output_parsers/openai_tools.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 7 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-export OpenAI tools output parsers from langchain_core for parsing structured tool call responses.

**Mechanism:** Imports and exposes JsonOutputKeyToolsParser, JsonOutputToolsParser, and PydanticToolsParser from langchain_core.output_parsers.openai_tools, providing access to parsers for OpenAI's newer tools API format.

**Significance:** Provides a convenient import location for parsers handling OpenAI's tools API (the evolution of function calling), enabling extraction of tool invocation data as JSON or Pydantic objects for agent-based workflows.
