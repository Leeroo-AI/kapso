# File: `libs/text-splitters/langchain_text_splitters/json.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 157 |
| Classes | `RecursiveJsonSplitter` |
| Imports | __future__, copy, json, langchain_core, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Splits JSON data into smaller chunks while preserving hierarchical structure and key-value relationships.

**Mechanism:** RecursiveJsonSplitter traverses JSON dictionaries recursively, measuring chunk size via json.dumps. When adding a key-value pair would exceed max_chunk_size, starts a new chunk (if current chunk meets min_chunk_size threshold). Uses _set_nested_dict to preserve the path structure in each chunk. Optionally converts lists to index-keyed dictionaries for better chunking via _list_to_dict_preprocessing.

**Significance:** Essential for RAG applications working with structured JSON data like API responses or configuration files. By preserving JSON hierarchy in chunks, it maintains the semantic context needed for accurate retrieval and prevents breaking related key-value pairs across chunk boundaries.
