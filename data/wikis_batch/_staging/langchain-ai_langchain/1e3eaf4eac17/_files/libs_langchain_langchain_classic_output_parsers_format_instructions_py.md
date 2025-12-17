# File: `libs/langchain/langchain_classic/output_parsers/format_instructions.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 79 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Define template strings for format instructions used across various output parsers to guide LLM output formatting.

**Mechanism:** Provides constant string templates including STRUCTURED_FORMAT_INSTRUCTIONS (JSON with markdown code blocks), PYDANTIC_FORMAT_INSTRUCTIONS (JSON schema-based), YAML_FORMAT_INSTRUCTIONS (YAML with examples), and PANDAS_DATAFRAME_FORMAT_INSTRUCTIONS (DataFrame query operations). Templates use placeholder substitution for dynamic schema/column information.

**Significance:** Centralizes format instruction templates to ensure consistent LLM prompting across different output parser types, reducing duplication and making it easier to maintain clear, standardized instructions for how LLMs should structure their responses.
