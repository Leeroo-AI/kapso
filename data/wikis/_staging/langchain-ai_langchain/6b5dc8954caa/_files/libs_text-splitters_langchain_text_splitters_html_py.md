# File: `libs/text-splitters/langchain_text_splitters/html.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1006 |
| Classes | `ElementType`, `HTMLHeaderTextSplitter`, `HTMLSectionSplitter`, `HTMLSemanticPreservingSplitter` |
| Imports | __future__, copy, io, langchain_core, langchain_text_splitters, pathlib, re, requests, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides specialized text splitters for HTML content that preserve document structure and semantic meaning.

**Mechanism:** HTMLHeaderTextSplitter uses BeautifulSoup to parse HTML and splits on specified header tags (h1-h6), tracking header hierarchy as metadata in Document objects. HTMLSectionSplitter uses XSLT transformations via lxml to convert tags to headers, then splits with RecursiveCharacterTextSplitter. HTMLSemanticPreservingSplitter (beta) preserves full HTML elements, converts links/media to Markdown format, supports custom handlers for specific tags, and includes optional stopword removal and text normalization.

**Significance:** These splitters enable structure-aware HTML processing for web scraping and documentation RAG use cases. The header hierarchy metadata allows retrieval systems to understand document organization and context, while semantic preservation maintains the relationship between content and its structural meaning.
