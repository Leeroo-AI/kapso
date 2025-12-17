# File: `libs/langchain/langchain_classic/output_parsers/regex_dict.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 42 |
| Classes | `RegexDictParser` |
| Imports | __future__, langchain_core, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parse LLM output into a dictionary by matching multiple key-value patterns using a configurable regex template.

**Mechanism:** Takes an output_key_to_format mapping and a regex_pattern template (default "{}:\s?([^.'\n']*)\.?"), searches for each expected format in the text, extracts values, validates exactly one match per key, and optionally skips keys matching no_update_value.

**Significance:** Enables extraction of multiple labeled fields from semi-structured LLM output where keys have consistent formatting patterns, useful for parsing outputs like "Name: John\nAge: 30\nCity: NYC" into structured dictionaries.
