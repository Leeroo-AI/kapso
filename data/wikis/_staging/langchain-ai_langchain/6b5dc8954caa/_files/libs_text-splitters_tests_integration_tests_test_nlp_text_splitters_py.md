# File: `libs/text-splitters/tests/integration_tests/test_nlp_text_splitters.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 123 |
| Functions | `setup_module`, `spacy`, `test_nltk_text_splitting_args`, `test_spacy_text_splitting_args`, `test_nltk_text_splitter`, `test_spacy_text_splitter`, `test_spacy_text_splitter_strip_whitespace`, `test_nltk_text_splitter_args`, `... +1 more` |
| Imports | langchain_core, langchain_text_splitters, nltk, pytest, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration tests for NLP-based text splitters (NLTK and Spacy) that split text on sentence boundaries, validating argument validation, splitting behavior, whitespace handling, and start index tracking.

**Mechanism:** Downloads NLTK punkt tokenizer in `setup_module()`. Uses pytest fixture to conditionally import Spacy and check for `en_core_web_sm` model availability. Tests cover: invalid argument validation (chunk_overlap > chunk_size), basic sentence splitting with custom separators, parametrized Spacy pipeline testing (sentencizer vs full model), whitespace preservation, span tokenization constraints, and document splitting with start index metadata.

**Significance:** Validates NLP-powered sentence-aware text splitting, which is critical for semantic chunking in RAG applications. These splitters preserve sentence boundaries to maintain semantic coherence, unlike character-based splitters. Essential for ensuring quality document chunking when semantic integrity matters more than character count.
