# File: `libs/langchain/langchain_classic/text_splitter.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 50 |
| Imports | langchain_text_splitters |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backwards compatibility re-export of text splitter utilities moved to dedicated package.

**Mechanism:** Directly imports and re-exports all text splitter classes from langchain_text_splitters package, including RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter, and other specialized splitters. No deprecation warnings since this is considered stable re-export.

**Significance:** Maintains import compatibility after text splitters were extracted into separate libs/text-splitters package, allowing legacy code to import from langchain.text_splitter while using the new package structure.
