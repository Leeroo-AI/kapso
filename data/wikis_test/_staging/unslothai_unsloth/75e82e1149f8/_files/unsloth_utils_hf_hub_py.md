# File: `unsloth/utils/hf_hub.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 78 |
| Functions | `formatted_int`, `get_model_info`, `list_models` |
| Imports | huggingface_hub |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides convenient wrappers around Hugging Face Hub API for retrieving model information and formatting popularity metrics.

**Mechanism:** Maintains singleton `_HFAPI` HfApi instance for connection reuse. `formatted_int()` converts integers to human-readable strings with K/M/B suffixes (e.g., 1500 -> "1.5K", 2000000 -> "2.0M"). `get_model_info()` fetches metadata for a specific model with configurable properties (defaults to "safetensors" and "lastModified" for minimal data transfer), with error handling that prints messages rather than raising exceptions. `list_models()` queries Hub for multiple models with filtering by author (default "unsloth"), search terms, sort order (default "downloads"), and limit (default 10), supporting both minimal property fetching and full model info retrieval.

**Significance:** Essential utility for model discovery and validation throughout Unsloth. Used by registry system to verify models exist on Hub (`_check_model_info()` in registry.py), and by DeepSeek registration to discover distilled model variants. The popularity formatting constants and function enable user-friendly display of download counts and likes. The singleton pattern for HfApi reduces connection overhead when making multiple Hub queries.
