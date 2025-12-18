# File: `libs/langchain/langchain_classic/output_parsers/datetime.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 58 |
| Classes | `DatetimeOutputParser` |
| Imports | datetime, langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parses LLM text output into Python datetime objects using configurable format strings.

**Mechanism:** Uses strptime with configurable format (defaults to ISO 8601 with microseconds). Generates format instructions with examples by formatting current time or using hardcoded examples for default format. Wraps ValueError in OutputParserException for consistent error handling.

**Significance:** Specialized parser for temporal data extraction from LLMs. Handles the common use case of getting structured date/time information from natural language or formatted text responses.
