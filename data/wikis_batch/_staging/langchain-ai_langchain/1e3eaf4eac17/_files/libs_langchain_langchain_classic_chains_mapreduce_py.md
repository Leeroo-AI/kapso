# File: `libs/langchain/langchain_classic/chains/mapreduce.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 117 |
| Classes | `MapReduceChain` |
| Imports | __future__, collections, langchain_classic, langchain_core, langchain_text_splitters, pydantic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Deprecated chain that splits large documents, processes chunks with an LLM (map), then combines results (reduce).

**Mechanism:** Accepts text input, splits it using a `TextSplitter`, converts chunks to `Document` objects, and passes them to a `MapReduceDocumentsChain` which applies an LLM chain to each chunk then reduces results using a combine chain (typically `StuffDocumentsChain` wrapped in `ReduceDocumentsChain`).

**Significance:** Classic distributed processing pattern for handling documents too large for a single LLM call. Deprecated since 0.2.13 in favor of LangGraph implementations which offer better control and flexibility. Historical importance for document summarization and analysis workflows.
