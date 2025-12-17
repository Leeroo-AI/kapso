# File: `libs/langchain/langchain_classic/output_parsers/enum.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 45 |
| Classes | `EnumOutputParser` |
| Imports | enum, langchain_core, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parse LLM text output into Python Enum values, enforcing selection from a predefined set of string options.

**Mechanism:** Takes an Enum class with string values, validates that all enum values are strings during initialization, strips and matches LLM output against valid enum values, and raises OutputParserException with helpful error messages if the response doesn't match any valid option.

**Significance:** Provides type-safe constrained choice parsing for LLM responses, ensuring outputs are valid enum members rather than arbitrary strings, which is critical for classification tasks and decision trees with fixed options.
