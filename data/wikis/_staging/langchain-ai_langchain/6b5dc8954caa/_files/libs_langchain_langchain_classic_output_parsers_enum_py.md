# File: `libs/langchain/langchain_classic/output_parsers/enum.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 45 |
| Classes | `EnumOutputParser` |
| Imports | enum, langchain_core, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parses LLM output to validate and convert text to predefined enum values.

**Mechanism:** Validates enum has only string values at initialization. Attempts to construct enum from stripped response text, raising OutputParserException if value not in enum. Provides format instructions listing all valid options.

**Significance:** Ensures LLM outputs are constrained to predefined choices, useful for classification tasks, menu selection, or any scenario requiring outputs from a fixed set of options.
