# File: `libs/langchain/langchain_classic/chains/loading.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 742 |
| Functions | `load_chain_from_config`, `load_chain` |
| Imports | __future__, json, langchain_classic, langchain_core, pathlib, typing, yaml |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides deprecated functionality for loading chain configurations from JSON/YAML files.

**Mechanism:** Maps chain type strings to loader functions via `type_to_loader_dict`. Each loader function (`_load_llm_chain`, `_load_retrieval_qa`, etc.) reconstructs chains from config dictionaries by deserializing components (prompts, LLMs, retrievers) and wiring them together. The `load_chain` function handles file I/O and delegates to `load_chain_from_config`.

**Significance:** Legacy serialization system for chains (deprecated since 0.2.13). While no longer recommended, it remains for backward compatibility with saved chain configurations. Some loaders (PALChain, SQLDatabaseChain) raise NotImplementedError for security reasons, directing users to langchain_experimental.
