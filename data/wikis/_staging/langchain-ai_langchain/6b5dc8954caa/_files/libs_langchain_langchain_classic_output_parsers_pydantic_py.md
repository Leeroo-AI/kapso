# File: `libs/langchain/langchain_classic/output_parsers/pydantic.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 3 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-exports PydanticOutputParser from langchain_core for backward compatibility.

**Mechanism:** Single-line import and re-export of PydanticOutputParser from langchain_core.output_parsers.

**Significance:** Minimal compatibility shim for one of the most important parsers - PydanticOutputParser enables type-safe structured output extraction using Pydantic models. Maintains classic import path for this heavily-used functionality.
