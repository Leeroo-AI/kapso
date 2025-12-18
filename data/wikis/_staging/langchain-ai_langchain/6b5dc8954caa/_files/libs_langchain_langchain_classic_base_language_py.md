# File: `libs/langchain/langchain_classic/base_language.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 7 |
| Imports | __future__, langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backwards compatibility shim for BaseLanguageModel class.

**Mechanism:** Simple re-export of `BaseLanguageModel` from `langchain_core.language_models`. No additional logic or transformation.

**Significance:** Maintains import compatibility for legacy code that imported `BaseLanguageModel` from langchain_classic. The actual implementation lives in langchain_core; this is just a convenience redirect.
