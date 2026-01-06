# File: `libs/text-splitters/tests/unit_tests/test_text_splitters.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 3881 |
| Functions | `test_character_text_splitter`, `test_character_text_splitter_empty_doc`, `test_character_text_splitter_separtor_empty_doc`, `test_character_text_splitter_long`, `test_character_text_splitter_short_words_first`, `test_character_text_splitter_longer_words`, `test_character_text_splitter_keep_separator_regex`, `test_character_text_splitter_keep_separator_regex_start`, `... +95 more` |
| Imports | __future__, langchain_core, langchain_text_splitters, pytest, random, re, string, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive test suite for all text splitting functionality across multiple languages and formats.

**Mechanism:** Tests character-based splitting (CharacterTextSplitter with overlap, separators, regex patterns), recursive splitting (RecursiveCharacterTextSplitter), language-specific splitters (Python, JavaScript, Go, Java, Kotlin, etc.), markdown/HTML header splitters (MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter, HTMLSectionSplitter), JSON splitting (RecursiveJsonSplitter), and semantic HTML splitting (HTMLSemanticPreservingSplitter). Uses pytest parametrization for testing multiple configurations and fixtures for reusable setup.

**Significance:** Critical test suite ensuring text splitters maintain correct behavior across 100+ test cases covering edge cases like empty documents, separator handling (keep/discard/position), code block preservation, metadata propagation, start_index tracking, and security features. Validates core document processing pipeline.
