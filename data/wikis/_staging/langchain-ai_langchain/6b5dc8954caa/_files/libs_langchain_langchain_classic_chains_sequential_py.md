# File: `libs/langchain/langchain_classic/chains/sequential.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 208 |
| Classes | `SequentialChain`, `SimpleSequentialChain` |
| Imports | langchain_classic, langchain_core, pydantic, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements sequential chain execution where outputs of one chain feed as inputs to the next.

**Mechanism:** `SequentialChain` validates that all required inputs are available (from initial inputs, memory, or previous chain outputs) and executes chains in order, accumulating outputs. `SimpleSequentialChain` is a simplified version requiring single-input/single-output chains, passing output directly as next chain's input with optional whitespace stripping.

**Significance:** Enables multi-step workflows and chain composition patterns. `SequentialChain` supports complex dataflows with multiple variables, while `SimpleSequentialChain` provides an intuitive interface for simple pipelines (e.g., generate draft -> critique -> revise).
