# File: `libs/langchain/langchain_classic/output_parsers/regex_dict.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 42 |
| Classes | `RegexDictParser` |
| Imports | __future__, langchain_core, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extracts dictionary of values from LLM output using key-specific regex patterns.

**Mechanism:** For each key in output_key_to_format, applies customized regex pattern (format string + regex template). Validates exactly one match per key. Optionally skips keys matching no_update_value. Returns dictionary of extracted values.

**Significance:** Specialized parser for labeled key-value extraction where each field has a specific format pattern. Useful for structured responses with labeled sections like "Name: ...", "Description: ...".
