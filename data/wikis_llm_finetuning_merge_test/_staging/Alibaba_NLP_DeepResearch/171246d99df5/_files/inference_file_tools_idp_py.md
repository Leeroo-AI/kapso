# File: `inference/file_tools/idp.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 90 |
| Classes | `IDP` |
| Imports | alibabacloud_credentials, alibabacloud_docmind_api20220711, alibabacloud_tea_openapi, alibabacloud_tea_util, json, os |

## Understanding

**Status:** Explored

**Purpose:** Integration wrapper for Alibaba Cloud's DocMind API (Intelligent Document Processing service) that provides cloud-based document parsing capabilities for PDFs and other document formats.

**Mechanism:** The `IDP` class initializes an Alibaba Cloud DocMind API client using API credentials from environment variables (`IDP_KEY_ID`, `IDP_KEY_SECRET`). It exposes three main methods: `file_submit_with_url()` submits a document URL for parsing, `file_submit_with_path()` uploads a local file for parsing, and `file_parser_query()` polls for parsing results and aggregates paginated layout data. The service returns markdown-formatted content organized by page.

**Significance:** Utility component that provides enhanced document parsing through Alibaba's cloud AI services. Acts as an optional high-quality parsing backend for the file_parser module, particularly useful for complex document layouts that require OCR or advanced structure recognition.
