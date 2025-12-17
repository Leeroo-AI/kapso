# File: `libs/text-splitters/langchain_text_splitters/markdown.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 468 |
| Classes | `MarkdownTextSplitter`, `MarkdownHeaderTextSplitter`, `LineType`, `HeaderType`, `ExperimentalMarkdownSyntaxTextSplitter` |
| Imports | __future__, langchain_core, langchain_text_splitters, re, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides Markdown-aware text splitting that preserves document structure through header hierarchy and code block boundaries.

**Mechanism:** MarkdownTextSplitter is a thin wrapper using Language.MARKDOWN separators (# headers, code blocks ```, horizontal rules, paragraphs). MarkdownHeaderTextSplitter is more sophisticated: parses line-by-line tracking header stack, detects standard (# Header) and custom header patterns (**Header**), maintains header hierarchy metadata, handles code blocks (``` and ~~~), and aggregates content with common headers using aggregate_lines_to_chunks(). Supports return_each_line mode and strip_headers option. ExperimentalMarkdownSyntaxTextSplitter (newer implementation) retains exact whitespace, uses regex matching for headers/code/horizontal rules, performs stateful parsing with header stack, and extracts code block languages into metadata. Both support custom_header_patterns for non-standard header formats.

**Significance:** Essential for documentation processing where header context is critical for retrieval. MarkdownHeaderTextSplitter is widely used for splitting README files, wikis, and technical docs while maintaining section context. The experimental version provides improved whitespace handling and code block extraction. Header-aware splitting dramatically improves RAG retrieval quality by preserving hierarchical relationships and allowing metadata filtering.
