# File: `libs/langchain/langchain_classic/chains/history_aware_retriever.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 68 |
| Functions | `create_history_aware_retriever` |
| Imports | __future__, langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Creates a retrieval chain that handles conversation history by reformulating queries when chat history exists.

**Mechanism:** Returns a `RunnableBranch` that checks for chat history. If absent, passes input directly to retriever. If present, uses an LLM with the provided prompt to reformulate the query based on history, then passes the reformulated query to the retriever.

**Significance:** Essential for conversational retrieval applications where follow-up questions reference previous context. Enables contextual query understanding without requiring users to manually reformulate standalone questions from conversational exchanges.
