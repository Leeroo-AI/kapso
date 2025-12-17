# File: `libs/text-splitters/langchain_text_splitters/html.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1006 |
| Classes | `ElementType`, `HTMLHeaderTextSplitter`, `HTMLSectionSplitter`, `HTMLSemanticPreservingSplitter` |
| Imports | __future__, copy, io, langchain_core, langchain_text_splitters, pathlib, re, requests, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides specialized HTML document splitting that preserves semantic structure through header hierarchy and element-based chunking.

**Mechanism:** HTMLHeaderTextSplitter parses HTML with BeautifulSoup, performs DFS traversal to extract text while tracking header hierarchy (h1-h6), and associates content with active headers as metadata. Supports split_text(), split_text_from_url(), and split_text_from_file() methods with optional return_each_element mode. HTMLSectionSplitter uses XSLT transformations (with lxml) to convert specific tags to headers before splitting, then applies RecursiveCharacterTextSplitter. HTMLSemanticPreservingSplitter (beta) is the most advanced, preserving full HTML elements with configurable max_chunk_size, optional media/link conversion to Markdown, custom tag handlers, stopword removal, text normalization, and allowlist/denylist filtering. It processes HTML hierarchically, preserves specific elements (tables, lists), and uses RecursiveCharacterTextSplitter for oversized chunks.

**Significance:** Essential for web scraping and document processing workflows where maintaining HTML structure and header context is critical for retrieval quality. HTMLSemanticPreservingSplitter represents the state-of-the-art approach for HTML chunking, balancing semantic preservation with size constraints. Used heavily in documentation ingestion and web content RAG systems.
