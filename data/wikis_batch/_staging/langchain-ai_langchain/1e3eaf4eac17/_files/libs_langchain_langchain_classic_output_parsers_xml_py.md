# File: `libs/langchain/langchain_classic/output_parsers/xml.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 3 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Re-export XMLOutputParser from langchain_core for parsing XML-formatted LLM output.

**Mechanism:** Imports and exposes XMLOutputParser from langchain_core.output_parsers.xml, which parses XML output from LLMs into structured data.

**Significance:** Provides a convenient import location for XML parsing capabilities, enabling extraction of hierarchical structured data from LLMs that output XML format, useful for legacy systems or specific domains where XML is the preferred data format.
