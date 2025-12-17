# File: `libs/langchain/tests/data.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 12 |
| Imports | pathlib |

## Understanding

**Status:** ✅ Explored

**Purpose:** Centralized test data file path definitions

**Mechanism:** Defines module-level constants for commonly used test PDF files (HELLO_PDF, LAYOUT_PARSER_PAPER_PDF, DUPLICATE_CHARS) located in the `integration_tests/examples/` directory, using Path objects for cross-platform compatibility.

**Significance:** Test utility that provides consistent, reusable file paths for integration tests requiring PDF fixtures. Centralizes test data locations to avoid hardcoded paths scattered throughout the test suite, making tests more maintainable and portable.
