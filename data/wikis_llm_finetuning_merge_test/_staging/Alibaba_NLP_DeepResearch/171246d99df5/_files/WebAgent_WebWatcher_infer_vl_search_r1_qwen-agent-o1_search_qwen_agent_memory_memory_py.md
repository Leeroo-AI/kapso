# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/memory/memory.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 137 |
| Classes | `Memory` |
| Imports | importlib, json, json5, qwen_agent, typing |

## Understanding

**Status:** Explored

**Purpose:** Implements a specialized Memory agent for file management and Retrieval-Augmented Generation (RAG) in the Qwen agent framework.

**Mechanism:** The `Memory` class extends `Agent` and provides document processing capabilities:
- Initializes with configurable RAG settings (max_ref_token, parser_page_size, rag_searchers, rag_keygen_strategy)
- Automatically configures `retrieval` and `doc_parser` tools for document handling
- The `_run()` method processes messages by: (1) extracting RAG-compatible files (PDF, DOCX, PPTX, TXT, HTML, CSV, TSV, XLSX, XLS), (2) generating keywords from user queries using configurable keygen strategies, (3) calling the retrieval tool to fetch relevant document segments
- The `get_rag_files()` helper filters files from messages to only include supported document types

**Significance:** Core component of the Qwen agent system that enables document-grounded conversations. It bridges file uploads with the retrieval system, allowing agents to answer questions based on uploaded documents. Essential for RAG-based workflows in the WebWatcher/vl_search system.
