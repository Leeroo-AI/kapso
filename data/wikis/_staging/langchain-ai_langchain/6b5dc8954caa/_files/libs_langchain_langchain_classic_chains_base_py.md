# File: `libs/langchain/langchain_classic/chains/base.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 806 |
| Classes | `Chain` |
| Imports | abc, builtins, contextlib, inspect, json, langchain_classic, langchain_core, logging, pathlib, pydantic, ... +4 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines the abstract `Chain` base class that all chain implementations must inherit from.

**Mechanism:** Extends `RunnableSerializable` with chain-specific functionality including memory integration, callback management, input/output validation, and both sync/async execution methods (`_call`, `_acall`, `invoke`, `ainvoke`). Provides lifecycle hooks for preparation (`prep_inputs`, `prep_outputs`) and supports serialization to JSON/YAML. Implements the deprecated `__call__` and `run` methods for backward compatibility.

**Significance:** Core abstraction for the entire chains subsystem. Every chain implementation (LLMChain, RetrievalQA, etc.) inherits from this class, making it the foundational interface that enables stateful, observable, and composable chain operations across LangChain.
