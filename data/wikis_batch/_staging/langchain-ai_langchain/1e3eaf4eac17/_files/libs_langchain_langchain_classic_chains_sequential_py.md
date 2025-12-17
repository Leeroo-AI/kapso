# File: `libs/langchain/langchain_classic/chains/sequential.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 208 |
| Classes | `SequentialChain`, `SimpleSequentialChain` |
| Imports | langchain_classic, langchain_core, pydantic, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements chains that execute multiple sub-chains in sequence, passing outputs to subsequent inputs.

**Mechanism:** `SequentialChain` accepts chains with multiple I/O variables, validates that each chain's inputs are satisfied by previous outputs, and accumulates results in a dictionary. `SimpleSequentialChain` is a simplified version requiring single-input/single-output chains, directly passing each output as the next input with optional string stripping.

**Significance:** Enables complex multi-stage processing pipelines where each chain builds on previous results. `SequentialChain` handles complex data flow with multiple variables, while `SimpleSequentialChain` provides a simpler linear pipeline. Both support async execution and proper callback management for observability.
