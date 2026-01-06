# File: `libs/langchain/langchain_classic/output_parsers/rail_parser.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 25 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Deprecated import shim for Guardrails AI output parser.

**Mechanism:** Uses create_importer with DEPRECATED_LOOKUP to dynamically redirect GuardrailsOutputParser imports to langchain_community.output_parsers.rail_parser. Provides __getattr__ hook and TYPE_CHECKING imports.

**Significance:** Backward compatibility for Guardrails AI integration (third-party validation framework). Part of moving third-party integrations to langchain_community package.
