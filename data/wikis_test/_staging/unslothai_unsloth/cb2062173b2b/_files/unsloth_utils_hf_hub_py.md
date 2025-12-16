# File: `unsloth/utils/hf_hub.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 78 |
| Functions | `formatted_int`, `get_model_info`, `list_models` |
| Imports | huggingface_hub |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides helper functions for interacting with the Hugging Face Hub API to retrieve model information and listings with optimized minimal data fetching.

**Mechanism:** Implements three main utilities:
- `formatted_int()`: Formats large integers into human-readable strings (e.g., 1500 -> "1.5K", 2500000 -> "2.5M")
- `get_model_info()`: Fetches metadata for a specific model with configurable properties to retrieve only needed fields (defaults to ["safetensors", "lastModified"]), uses singleton HfApi instance
- `list_models()`: Queries Hub for models matching criteria (author, search, sort order) with optional full or minimal property expansion
Maintains a global _HFAPI singleton to avoid repeated API client initialization. Includes error handling for failed API requests.

**Significance:** Utility module for model discovery and metadata retrieval from Hugging Face Hub. Particularly useful for checking model availability, formats (safetensors), modification dates, and popularity metrics. The default minimal property fetching improves performance when only basic information is needed, while the full mode allows comprehensive metadata retrieval when required.
