# File: `libs/text-splitters/tests/integration_tests/test_nlp_text_splitters.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 123 |
| Functions | `setup_module`, `spacy`, `test_nltk_text_splitting_args`, `test_spacy_text_splitting_args`, `test_nltk_text_splitter`, `test_spacy_text_splitter`, `test_spacy_text_splitter_strip_whitespace`, `test_nltk_text_splitter_args`, `... +1 more` |
| Imports | langchain_core, langchain_text_splitters, nltk, pytest, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests NLTK and Spacy-based text splitters that use NLP models for sentence-aware text splitting.

**Mechanism:** Downloads NLTK punkt tokenizer in `setup_module`, uses a pytest fixture to conditionally skip Spacy tests if the en_core_web_sm model isn't installed. Tests validate argument validation, basic splitting functionality, separator handling, whitespace preservation, and start_index metadata tracking for both NLTKTextSplitter and SpacyTextSplitter with different pipeline configurations.

**Significance:** Integration tests requiring external NLP dependencies. Verifies that text splitters correctly leverage NLTK and Spacy for sentence-based splitting, which is more sophisticated than character-based splitting. These splitters are crucial for maintaining semantic boundaries when chunking documents.
