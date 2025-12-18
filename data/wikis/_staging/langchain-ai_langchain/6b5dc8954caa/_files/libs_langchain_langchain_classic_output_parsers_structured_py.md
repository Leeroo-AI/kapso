# File: `libs/langchain/langchain_classic/output_parsers/structured.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 116 |
| Classes | `ResponseSchema`, `StructuredOutputParser` |
| Imports | __future__, langchain_classic, langchain_core, pydantic, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parses LLM output into structured dictionaries based on predefined schema with field names, types, and descriptions.

**Mechanism:** Uses ResponseSchema to define expected fields (name, description, type). Generates format instructions as markdown JSON code block with field annotations. Parses output using parse_and_check_json_markdown to validate expected keys are present.

**Significance:** Lightweight alternative to PydanticOutputParser when full Pydantic models are unnecessary. Ideal for simple structured extraction with basic type information without needing class definitions.
