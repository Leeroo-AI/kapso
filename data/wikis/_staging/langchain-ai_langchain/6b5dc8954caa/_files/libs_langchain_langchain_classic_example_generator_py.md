# File: `libs/langchain/langchain_classic/example_generator.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 5 |
| Imports | langchain_classic |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backwards compatibility shim for example generation functionality.

**Mechanism:** Simple re-export of `generate_example` from `langchain_classic.chains.example_generator`. No additional logic.

**Significance:** Maintains import compatibility for legacy code that imported `generate_example` directly from this module path. The actual implementation has been moved to the chains subpackage, but this redirect preserves the old import path.
