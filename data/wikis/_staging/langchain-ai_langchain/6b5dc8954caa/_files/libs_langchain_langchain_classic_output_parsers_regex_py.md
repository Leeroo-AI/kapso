# File: `libs/langchain/langchain_classic/output_parsers/regex.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 40 |
| Classes | `RegexParser` |
| Imports | __future__, langchain_core, re, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extracts structured data from LLM output using regular expressions with named capture groups.

**Mechanism:** Applies configured regex pattern to text, maps capture groups to output_keys sequentially. If no match and default_output_key set, returns entire text for that key with empty strings for others. Raises ValueError on no match without default.

**Significance:** Flexible pattern-based extraction for custom output formats. Useful when LLM outputs follow predictable patterns not covered by standard parsers. Serializable for persistence.
