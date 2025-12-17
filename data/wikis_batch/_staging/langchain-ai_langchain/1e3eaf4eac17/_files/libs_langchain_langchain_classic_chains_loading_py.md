# File: `libs/langchain/langchain_classic/chains/loading.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 742 |
| Functions | `load_chain_from_config`, `load_chain` |
| Imports | __future__, json, langchain_classic, langchain_core, pathlib, typing, yaml |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides deprecated functionality for loading chain configurations from JSON/YAML files or dictionaries.

**Mechanism:** Maintains a `type_to_loader_dict` mapping chain types to loader functions. Each loader function reconstructs a chain from config by loading sub-components (LLMs, prompts, sub-chains) and passing them to chain constructors. Supports loading from files (`load_chain`) or config dicts (`load_chain_from_config`).

**Significance:** Legacy serialization/deserialization system that enabled saving and loading chain configurations. Deprecated since 0.2.13 in favor of directly importing chains. Includes loaders for various chain types (LLMChain, MapReduce, QA chains, etc.) with some chains like SQLDatabaseChain moved to experimental packages for security reasons.
