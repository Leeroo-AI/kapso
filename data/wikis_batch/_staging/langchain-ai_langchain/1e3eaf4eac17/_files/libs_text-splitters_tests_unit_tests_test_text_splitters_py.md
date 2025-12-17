# File: `libs/text-splitters/tests/unit_tests/test_text_splitters.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 3881 |
| Functions | `test_character_text_splitter`, `test_character_text_splitter_empty_doc`, `test_character_text_splitter_separtor_empty_doc`, `test_character_text_splitter_long`, `test_character_text_splitter_short_words_first`, `test_character_text_splitter_longer_words`, `test_character_text_splitter_keep_separator_regex`, `test_character_text_splitter_keep_separator_regex_start`, `... +95 more` |
| Imports | __future__, langchain_core, langchain_text_splitters, pytest, random, re, string, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive unit test suite covering all text splitter implementations in the langchain-text-splitters package, including character-based, recursive, language-specific, markdown, HTML, and JSON splitters.

**Mechanism:** Contains 103+ test functions organized by splitter type. Tests character-based splitting with various separators (regex and literal), chunk sizes, overlaps, and separator handling modes (keep/discard, start/end). Validates language-specific code splitters (Python, JavaScript, Java, Go, Rust, etc.), markdown header splitting with custom patterns, HTML header/section splitting with BeautifulSoup, JSON recursive splitting, and the experimental HTMLSemanticPreservingSplitter with extensive edge cases. Uses parametrized tests, fixtures, and comprehensive assertions.

**Significance:** The primary test suite ensuring correctness of all text splitting functionality. Critical for validating that documents are split appropriately for LLM context windows while preserving semantic boundaries. Tests cover edge cases like empty documents, whitespace handling, metadata preservation, security concerns, and language-specific syntax awareness across 20+ programming languages and document formats.
