# File: `libs/langchain/langchain_classic/output_parsers/regex.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 40 |
| Classes | `RegexParser` |
| Imports | __future__, langchain_core, re, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parse LLM output using custom regex patterns to extract named groups into a dictionary.

**Mechanism:** Takes a regex pattern and list of output_keys, searches the LLM output text with the regex, extracts matched groups into a dictionary mapping keys to captured values, and optionally falls back to returning the full text for a default_output_key if no match is found.

**Significance:** Provides flexible custom parsing for LLM outputs with specific patterns, enabling extraction of structured data from semi-structured text when standard parsers don't fit the format, useful for domain-specific output formats.
