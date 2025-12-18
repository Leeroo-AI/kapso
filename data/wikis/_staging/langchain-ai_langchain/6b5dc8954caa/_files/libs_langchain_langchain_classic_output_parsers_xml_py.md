# File: `libs/langchain/langchain_classic/output_parsers/xml.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 3 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-exports XMLOutputParser from langchain_core for backward compatibility.

**Mechanism:** Single-line import and re-export of XMLOutputParser from langchain_core.output_parsers.

**Significance:** Compatibility shim for XML parsing functionality. Maintains classic import path for applications extracting structured data from XML-formatted LLM outputs.
