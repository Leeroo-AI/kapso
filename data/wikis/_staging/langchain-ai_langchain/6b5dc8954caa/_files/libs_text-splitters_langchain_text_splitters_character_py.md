# File: `libs/text-splitters/langchain_text_splitters/character.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 793 |
| Classes | `CharacterTextSplitter`, `RecursiveCharacterTextSplitter` |
| Imports | __future__, langchain_text_splitters, re, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements character-based text splitting with support for simple and recursive splitting strategies.

**Mechanism:** CharacterTextSplitter splits on a single separator (default "\n\n") using regex with lookaround support. RecursiveCharacterTextSplitter tries multiple separators hierarchically, splitting on the first match and recursing on oversized chunks. Contains get_separators_for_language with predefined separator lists for 30+ languages (Python, JavaScript, Java, C++, etc.) that split on language-specific constructs like function definitions and control flow statements.

**Significance:** RecursiveCharacterTextSplitter is the most commonly used general-purpose text splitter in LangChain. The language-specific separator definitions enable code-aware splitting that respects syntax boundaries, making it essential for RAG applications processing source code.
