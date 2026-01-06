# File: `libs/langchain/langchain_classic/formatting.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 5 |
| Imports | langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backwards compatibility shim for string formatting utilities.

**Mechanism:** Re-exports `StrictFormatter` and `formatter` from `langchain_core.utils.formatting`. No additional logic.

**Significance:** Maintains import compatibility for legacy code that used formatting utilities from langchain_classic. The implementations have been moved to langchain_core (the base abstraction layer), but this redirect preserves the old import path for backwards compatibility.
