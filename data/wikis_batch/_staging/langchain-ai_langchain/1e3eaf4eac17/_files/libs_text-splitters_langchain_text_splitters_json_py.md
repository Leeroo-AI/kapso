# File: `libs/text-splitters/langchain_text_splitters/json.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 157 |
| Classes | `RecursiveJsonSplitter` |
| Imports | __future__, copy, json, langchain_core, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Splits JSON data into smaller chunks while preserving hierarchical structure and ensuring chunks fit within size constraints.

**Mechanism:** Recursively traverses JSON dictionaries, measuring serialized size of key-value pairs using _json_size() and adding them to chunks based on max_chunk_size (default 2000) and min_chunk_size (default max-200, minimum 50). Uses _set_nested_dict() to preserve nested paths in output chunks. Optional convert_lists mode transforms arrays into dictionaries with index-based keys for more granular splitting. Provides split_json() returning dict chunks, split_text() returning JSON strings, and create_documents() for Document objects with metadata.

**Significance:** Specialized splitter for structured data that maintains JSON validity and hierarchical relationships across chunks. Critical for RAG systems working with JSON APIs, configuration files, or structured datasets where preserving object relationships is essential. The hierarchical splitting ensures related data stays together while respecting size limits.
