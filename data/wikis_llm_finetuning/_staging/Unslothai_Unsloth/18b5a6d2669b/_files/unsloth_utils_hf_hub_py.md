# File: `unsloth/utils/hf_hub.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 80 |
| Functions | `formatted_int`, `get_model_info`, `list_models` |
| Imports | huggingface_hub |

## Understanding

**Status:** ✅ Explored

**Purpose:** HuggingFace Hub integration utilities for model discovery and metadata retrieval

**Mechanism:** Wraps HfApi to provide get_model_info() for fetching individual model metadata and list_models() for querying model listings with filtering/sorting, includes formatted_int() helper for human-readable number display (K/M/B suffixes)

**Significance:** Enables Unsloth to programmatically interact with the HuggingFace Hub for model management, particularly useful for discovering and validating Unsloth-optimized models and displaying download statistics
