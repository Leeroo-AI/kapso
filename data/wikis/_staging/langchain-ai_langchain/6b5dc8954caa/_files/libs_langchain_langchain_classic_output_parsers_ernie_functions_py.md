# File: `libs/langchain/langchain_classic/output_parsers/ernie_functions.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 45 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Deprecated import shim for ERNIE (Baidu) function calling parsers.

**Mechanism:** Uses create_importer with DEPRECATED_LOOKUP to dynamically redirect imports to langchain_community.output_parsers.ernie_functions. Provides __getattr__ hook for lazy loading and TYPE_CHECKING imports for type hints.

**Significance:** Backward compatibility layer for code using ERNIE function parsers from the old location. Part of the reorganization moving vendor-specific parsers to langchain_community package.
