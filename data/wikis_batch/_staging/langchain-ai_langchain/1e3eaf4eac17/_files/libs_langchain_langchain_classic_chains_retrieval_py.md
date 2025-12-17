# File: `libs/langchain/langchain_classic/chains/retrieval.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 68 |
| Functions | `create_retrieval_chain` |
| Imports | __future__, langchain_core, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Creates a modern LCEL-based retrieval chain that retrieves documents and combines them with a processing chain.

**Mechanism:** Wraps a retriever (extracting from `input` key if BaseRetriever) and uses `RunnablePassthrough.assign` to add retrieved documents as `context`, then passes everything to a `combine_docs_chain` which produces an `answer`. Returns a Runnable that outputs both context and answer.

**Significance:** Core building block for retrieval-augmented generation (RAG) applications. Unlike deprecated chain classes, this uses the modern LCEL syntax for better composability. Standard pattern for question-answering systems that combine document retrieval with LLM-based answer generation.
