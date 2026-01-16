# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/simple_doc_parser.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 550 |
| Classes | `DocParserError`, `SimpleDocParser` |
| Functions | `parse_file_by_idp`, `clean_paragraph`, `parse_word`, `parse_ppt`, `parse_txt`, `df_to_md`, `parse_excel`, `parse_csv`, `... +8 more` |
| Imports | collections, json, os, qwen_agent, re, requests, time, typing |

## Understanding

**Status:** Explored

**Purpose:** Provides comprehensive document parsing capabilities to extract structured content from multiple file formats (PDF, DOCX, PPTX, TXT, HTML, CSV, TSV, XLSX/XLS), enabling LLM agents to process and analyze documents.

**Mechanism:** The `SimpleDocParser` class (registered as `'simple_doc_parser'`) implements format-specific parsers: (1) `parse_pdf()` uses pdfminer for text extraction and pdfplumber for table extraction, with font-based paragraph merging logic; (2) `parse_word()` uses python-docx to extract paragraphs and tables; (3) `parse_ppt()` uses python-pptx for slide content; (4) `parse_excel()`/`parse_csv()`/`parse_tsv()` use pandas with markdown table formatting via `df_to_md()`; (5) `parse_html_bs()` uses BeautifulSoup for web content; (6) Optional Alibaba IDP (Intelligent Document Processing) cloud service integration via `parse_file_by_idp()` for advanced PDF parsing. Features include: URL downloading and local file caching via `Storage`, token counting per paragraph, content safety inspection via `csi()`, and output in either plain text or structured format (page numbers, text/table content). Error handling via `DocParserError`.

**Significance:** Essential document understanding tool that bridges the gap between raw documents and LLM-processable text. Enables agents to answer questions about uploaded files, extract information from reports, and analyze structured data in spreadsheets. The caching mechanism improves efficiency for repeated document access in multi-turn conversations.
