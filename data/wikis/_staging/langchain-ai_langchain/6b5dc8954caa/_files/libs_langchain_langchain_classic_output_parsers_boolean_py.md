# File: `libs/langchain/langchain_classic/output_parsers/boolean.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 54 |
| Classes | `BooleanOutputParser` |
| Imports | langchain_core, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parses LLM text output to boolean values based on configurable string patterns.

**Mechanism:** Uses regex with word boundaries to search for configurable true/false values (defaults: "YES"/"NO") in text, case-insensitively. Validates that only one value is present to avoid ambiguity, raising ValueError if both appear or neither appears.

**Significance:** Specialized parser for binary decision tasks where LLMs need to produce yes/no, true/false outcomes. Handles common edge cases like ambiguous responses and missing values with clear error messages.
