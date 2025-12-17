# File: `unsloth/utils/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 48 |
| Imports | attention_dispatch, packing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Export public API for attention and packing utilities, providing unified access to attention backends and data packing functions.

**Mechanism:** Re-exports core functions from packing module (sample packing, padding-free configuration) and attention_dispatch module (attention backend selection and execution) with comprehensive __all__ list.

**Significance:** Provides clean public interface for critical performance optimization utilities used throughout the training pipeline.
