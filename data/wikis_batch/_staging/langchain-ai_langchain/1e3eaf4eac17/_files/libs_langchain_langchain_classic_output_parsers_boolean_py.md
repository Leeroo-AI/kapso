# File: `libs/langchain/langchain_classic/output_parsers/boolean.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 54 |
| Classes | `BooleanOutputParser` |
| Imports | langchain_core, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parse LLM text output into boolean values by matching configurable true/false string patterns.

**Mechanism:** Uses regex to search for configurable true_val (default "YES") or false_val (default "NO") strings in LLM output. Performs case-insensitive word boundary matching, raises errors on ambiguous responses containing both values or neither value, and returns the appropriate boolean result.

**Significance:** Provides a simple but robust way to convert free-form LLM text responses into boolean values for decision-making workflows, with built-in validation to prevent ambiguous or invalid responses.
