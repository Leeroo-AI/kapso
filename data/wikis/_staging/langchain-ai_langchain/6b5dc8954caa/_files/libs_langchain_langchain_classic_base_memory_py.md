# File: `libs/langchain/langchain_classic/base_memory.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 116 |
| Classes | `BaseMemory` |
| Imports | __future__, abc, langchain_core, pydantic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines the abstract base class for memory components in LangChain chains (deprecated as of v0.3.3).

**Mechanism:** `BaseMemory` is an abstract class that extends `Serializable` and defines the contract for memory implementations: `memory_variables` property (returns keys to inject), `load_memory_variables()` (retrieves context), `save_context()` (persists interaction), and `clear()` (resets memory). Includes async variants of each method using `run_in_executor()` for compatibility.

**Significance:** Core abstraction from LangChain v0.0.x for maintaining conversational context and chain state. Now deprecated in favor of newer memory patterns (see migration guide at python.langchain.com/docs/versions/migrating_memory/). Still present to support legacy chains and provide a clear interface for what memory systems should do.
