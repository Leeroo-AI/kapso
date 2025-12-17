# File: `libs/text-splitters/langchain_text_splitters/character.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 793 |
| Classes | `CharacterTextSplitter`, `RecursiveCharacterTextSplitter` |
| Imports | __future__, langchain_text_splitters, re, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements character-based text splitting strategies, including simple separator-based splitting and recursive multi-separator splitting with language-specific separators.

**Mechanism:** CharacterTextSplitter splits text on a single separator (default "\n\n") using regex, with support for lookaround patterns and separator retention. RecursiveCharacterTextSplitter tries multiple separators in order (e.g., ["\n\n", "\n", " ", ""]) to find natural split points, recursively splitting chunks that exceed chunk_size. Provides from_language() factory method and get_separators_for_language() static method with predefined separator hierarchies for 30+ languages (Python, JavaScript, Java, Go, Rust, etc.). Each language has carefully ordered separators prioritizing logical boundaries like class definitions, function declarations, and control flow statements before falling back to paragraph and word boundaries. Helper function _split_text_with_regex() handles separator retention logic.

**Significance:** Core splitting implementation used across LangChain for general text and code. RecursiveCharacterTextSplitter is the most commonly used splitter due to its intelligent fallback behavior that preserves semantic boundaries. Language-specific separators enable context-aware code splitting that maintains syntactic structure, crucial for RAG systems working with codebases.
