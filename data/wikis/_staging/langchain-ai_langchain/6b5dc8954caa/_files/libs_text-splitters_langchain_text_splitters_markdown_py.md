# File: `libs/text-splitters/langchain_text_splitters/markdown.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 468 |
| Classes | `MarkdownTextSplitter`, `MarkdownHeaderTextSplitter`, `LineType`, `HeaderType`, `ExperimentalMarkdownSyntaxTextSplitter` |
| Imports | __future__, langchain_core, langchain_text_splitters, re, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides multiple strategies for splitting Markdown documents while preserving structure and metadata.

**Mechanism:** MarkdownTextSplitter uses RecursiveCharacterTextSplitter with Markdown-specific separators (headings, code blocks, horizontal rules). MarkdownHeaderTextSplitter parses headers line-by-line, maintains a header stack for hierarchy tracking, handles code blocks, and attaches header metadata to content chunks. Supports custom header patterns and optional header stripping. ExperimentalMarkdownSyntaxTextSplitter is a rewrite that retains exact whitespace, extracts headers/code/horizontal rules as metadata, and includes code language in metadata.

**Significance:** Critical for documentation and knowledge base RAG applications. The header tracking metadata enables hierarchical understanding of content, while code block awareness prevents breaking syntax across chunks. The experimental version offers stricter whitespace preservation for use cases requiring exact formatting.
