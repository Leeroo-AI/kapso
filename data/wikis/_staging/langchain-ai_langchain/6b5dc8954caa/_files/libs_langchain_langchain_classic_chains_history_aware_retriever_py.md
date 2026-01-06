# File: `libs/langchain/langchain_classic/chains/history_aware_retriever.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 68 |
| Functions | `create_history_aware_retriever` |
| Imports | __future__, langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Creates a retriever chain that reformulates queries based on conversation history.

**Mechanism:** The `create_history_aware_retriever` function returns a `RunnableBranch` that conditionally processes inputs: if no `chat_history` exists, it passes the `input` directly to the retriever; otherwise, it uses an LLM with a prompt to reformulate the query considering conversation context before retrieving documents.

**Significance:** Essential component for conversational RAG systems. Enables contextually-aware document retrieval by allowing the LLM to rewrite user queries that reference previous conversation turns (e.g., "What about that?" becomes "What are the specifications of the iPhone 14?").
