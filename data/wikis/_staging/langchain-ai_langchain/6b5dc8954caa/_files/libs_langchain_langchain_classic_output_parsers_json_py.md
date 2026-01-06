# File: `libs/langchain/langchain_classic/output_parsers/json.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 15 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-exports JSON parsing utilities from langchain_core for backward compatibility.

**Mechanism:** Imports and re-exports SimpleJsonOutputParser and JSON utility functions (parse_and_check_json_markdown, parse_json_markdown, parse_partial_json) from langchain_core.

**Significance:** Compatibility shim maintaining the classic import path for JSON parsing functionality now implemented in langchain_core. Allows existing code to continue working without import changes.
