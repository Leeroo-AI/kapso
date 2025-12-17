# File: `unsloth/utils/hf_hub.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 78 |
| Functions | `formatted_int`, `get_model_info`, `list_models` |
| Imports | huggingface_hub |

## Understanding

**Status:** ✅ Explored

**Purpose:** Wrapper utilities for HuggingFace Hub API enabling efficient model discovery and metadata retrieval with popularity metrics formatting.

**Mechanism:** Provides get_model_info() for fetching metadata on specific models with customizable property expansion; list_models() for querying by author/search/sort with pagination; formatted_int() helper for human-readable metrics (K, M, B suffixes); uses singleton HfApi instance to avoid repeated initialization.

**Significance:** Abstracts HuggingFace Hub interactions with caching strategy and convenient filtering, supporting model discovery and popularity-based analysis.
