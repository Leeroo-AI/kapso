# File: `libs/langchain/langchain_classic/chains/mapreduce.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 117 |
| Classes | `MapReduceChain` |
| Imports | __future__, collections, langchain_classic, langchain_core, langchain_text_splitters, pydantic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the deprecated `MapReduceChain` for processing documents via map-reduce pattern.

**Mechanism:** Splits input text using a `TextSplitter`, converts chunks to `Document` objects, then applies a `MapReduceDocumentsChain` (which uses `LLMChain` for mapping and `ReduceDocumentsChain` for reduction). The `from_params` class method provides a convenient constructor from LLM, prompt, and text splitter.

**Significance:** Historical pattern for processing large documents that exceed context limits (deprecated since 0.2.13 in favor of LangGraph implementations). While superseded, it demonstrates the classic map-reduce approach: split text, process chunks in parallel, then combine results.
