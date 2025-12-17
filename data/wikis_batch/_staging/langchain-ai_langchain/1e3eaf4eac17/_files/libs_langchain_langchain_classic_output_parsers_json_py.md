# File: `libs/langchain/langchain_classic/output_parsers/json.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 15 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-export JSON parsing utilities from langchain_core for parsing and validating JSON output from LLMs.

**Mechanism:** Imports and exposes SimpleJsonOutputParser class and three utility functions (parse_and_check_json_markdown, parse_json_markdown, parse_partial_json) from langchain_core, providing a convenience layer for accessing JSON parsing functionality.

**Significance:** Provides a stable import location in langchain_classic for JSON parsing capabilities that are implemented in langchain_core, maintaining API consistency while allowing core functionality to remain in the base layer.
