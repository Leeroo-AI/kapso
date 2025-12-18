# File: `libs/langchain/langchain_classic/output_parsers/format_instructions.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 79 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Template strings for instructing LLMs on required output formats across different parser types.

**Mechanism:** Defines constant string templates with placeholders for structured JSON, Pydantic models, YAML, and Pandas DataFrame operations. Templates include examples, schema descriptions, and formatting requirements with markdown code blocks.

**Significance:** Centralized format instruction templates ensure consistent prompting across different parsers. Critical for LLM output quality as clear formatting instructions directly impact parsing success rates.
