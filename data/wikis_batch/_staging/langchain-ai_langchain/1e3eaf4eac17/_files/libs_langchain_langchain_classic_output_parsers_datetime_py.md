# File: `libs/langchain/langchain_classic/output_parsers/datetime.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 58 |
| Classes | `DatetimeOutputParser` |
| Imports | datetime, langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parse LLM text output into Python datetime objects using configurable format strings.

**Mechanism:** Uses strptime with a configurable format string (default ISO 8601 format "%Y-%m-%dT%H:%M:%S.%fZ") to parse text into datetime objects. Generates format instructions with example datetime strings based on the configured format, and raises OutputParserException on parsing failures.

**Significance:** Provides a standardized way to extract temporal information from LLM responses, essential for scheduling, logging, and time-based applications that require structured datetime data rather than free-form text.
