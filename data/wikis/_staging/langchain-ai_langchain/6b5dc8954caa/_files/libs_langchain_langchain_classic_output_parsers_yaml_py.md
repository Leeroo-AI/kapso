# File: `libs/langchain/langchain_classic/output_parsers/yaml.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 69 |
| Classes | `YamlOutputParser` |
| Imports | json, langchain_classic, langchain_core, pydantic, re, typing, typing_extensions, yaml |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parses YAML-formatted LLM output into Pydantic model instances.

**Mechanism:** Uses regex to extract YAML from triple backtick code blocks (with optional yaml/yml prefix). Falls back to parsing entire text if no code blocks found. Parses with yaml.safe_load, then validates against Pydantic model. Generates format instructions from model's JSON schema with examples.

**Significance:** Alternative to JSON parsing for LLM outputs. YAML's readability and support for comments can produce better LLM outputs in some cases. Provides type safety through Pydantic validation.
