# File: `libs/langchain/tests/data.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 12 |
| Imports | pathlib |

## Understanding

**Status:** ✅ Explored

**Purpose:** Centralizes test data file paths for integration tests in the legacy langchain package.

**Mechanism:** Uses pathlib to define constants pointing to example PDF files (HELLO_PDF, LAYOUT_PARSER_PAPER_PDF, DUPLICATE_CHARS) in the integration_tests/examples directory relative to the module location.

**Significance:** Provides a single source of truth for test data paths, preventing hardcoded paths throughout the test suite and enabling easy relocation of test fixtures if needed.
