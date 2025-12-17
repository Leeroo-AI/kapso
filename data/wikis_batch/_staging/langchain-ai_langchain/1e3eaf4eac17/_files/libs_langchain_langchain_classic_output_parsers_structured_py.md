# File: `libs/langchain/langchain_classic/output_parsers/structured.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 116 |
| Classes | `ResponseSchema`, `StructuredOutputParser` |
| Imports | __future__, langchain_classic, langchain_core, pydantic, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parse LLM output into structured dictionaries based on schema definitions without requiring Pydantic models.

**Mechanism:** Defines ResponseSchema with name, description, and type fields to specify expected outputs. StructuredOutputParser takes a list of ResponseSchema objects, generates format instructions as JSON markdown templates, parses LLM output using parse_and_check_json_markdown, and validates that all expected keys are present.

**Significance:** Provides a lightweight alternative to PydanticOutputParser for structured parsing when full Pydantic models are unnecessary, enabling simple schema-based parsing with clear format instructions and validation.
