# File: `libs/text-splitters/langchain_text_splitters/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 68 |
| Imports | langchain_text_splitters |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization file that exports all text splitter classes and utilities, serving as the public API for the langchain_text_splitters module.

**Mechanism:** Imports all text splitter implementations from their respective modules (base, character, html, json, jsx, konlpy, latex, markdown, nltk, python, sentence_transformers, spacy) and defines __all__ to explicitly control which symbols are exported when using `from langchain_text_splitters import *`. The file organizes exports alphabetically including splitter classes (CharacterTextSplitter, RecursiveCharacterTextSplitter, etc.), enums (Language, ElementType), and utility functions (split_text_on_tokens). Contains a docstring note that MarkdownHeaderTextSplitter and HTMLHeaderTextSplitter do not derive from TextSplitter.

**Significance:** Critical entry point that defines the public API surface for text splitting functionality. Provides centralized access to 20+ text splitter implementations, making them easily discoverable and importable. The __all__ list ensures only intended public APIs are exposed, maintaining clean package boundaries.
