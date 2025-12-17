# File: `libs/text-splitters/langchain_text_splitters/python.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 17 |
| Classes | `PythonCodeTextSplitter` |
| Imports | __future__, langchain_text_splitters, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Splits Python source code along Python-specific syntax boundaries to preserve code structure and semantic meaning.

**Mechanism:** Thin wrapper around RecursiveCharacterTextSplitter that retrieves Python language separators using get_separators_for_language(Language.PYTHON). The separator hierarchy (defined in character.py) prioritizes: class definitions ("\nclass "), function definitions ("\ndef ", "\n\tdef "), then generic text separators ("\n\n", "\n", " ", ""). Inherits all RecursiveCharacterTextSplitter functionality including recursive splitting and chunk size management.

**Significance:** Specialized code splitter that maintains Python's syntactic structure when chunking codebases for RAG. By splitting on class and function boundaries, it ensures code chunks are semantically meaningful and complete. Essential for code understanding, documentation generation, and code search applications where preserving function/class scope is critical for context.
