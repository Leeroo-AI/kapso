# File: `libs/langchain/langchain_classic/output_parsers/yaml.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 69 |
| Classes | `YamlOutputParser` |
| Imports | json, langchain_classic, langchain_core, pydantic, re, typing, typing_extensions, yaml |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parse YAML-formatted LLM output into Pydantic model instances with schema validation.

**Mechanism:** Takes a Pydantic model class, uses regex to extract YAML from triple backticks (with optional yaml/yml prefix) or parses entire output if no backticks found, loads YAML with yaml.safe_load, validates against the Pydantic model schema, and generates format instructions with JSON schema examples converted to YAML.

**Significance:** Provides YAML parsing as an alternative to JSON for LLM outputs, offering a more human-readable format that some LLMs may handle better, while maintaining type safety through Pydantic validation.
