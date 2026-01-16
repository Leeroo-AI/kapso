# File: `inference/file_tools/file_parser.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 578 |
| Classes | `CustomJSONEncoder`, `FileParserError`, `SingleFileParser` |
| Functions | `str_to_bool`, `parse_file_by_idp`, `process_idp_result`, `clean_text`, `get_plain_doc`, `df_to_markdown`, `parse_word`, `parse_ppt`, `... +14 more` |
| Imports | collections, datetime, file_tools, json, math, os, pandas, pathlib, qwen_agent, re, ... +5 more |

## Understanding

**Status:** Explored

**Purpose:** Comprehensive document parsing tool that extracts text content from multiple file formats (PDF, DOCX, PPTX, TXT, HTML, CSV, TSV, XLSX, XLS, ZIP, XML) and converts them to markdown format for processing by the research agent.

**Mechanism:** The `SingleFileParser` class extends `BaseTool` and uses format-specific parsers: pdfminer/pdfplumber for PDFs, python-docx for Word documents, python-pptx for presentations, pandas for tabular data, and BeautifulSoup for HTML. It supports both local files and HTTP URLs, with caching via SHA256 hashing. Optionally integrates with Alibaba's IDP (Intelligent Document Processing) service for enhanced parsing of PDFs, DOCX, PPTX, and XLSX files. Large tabular files are converted to schema representations when exceeding token limits.

**Significance:** Core component that enables the DeepResearch agent to process user-uploaded documents. It serves as the document ingestion layer, converting diverse file formats into a standardized text representation that can be used for information extraction and analysis in the research pipeline.
