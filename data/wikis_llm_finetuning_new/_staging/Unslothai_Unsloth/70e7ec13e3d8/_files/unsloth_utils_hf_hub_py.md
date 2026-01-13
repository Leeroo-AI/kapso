# File: `unsloth/utils/hf_hub.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 80 |
| Functions | `formatted_int`, `get_model_info`, `list_models` |
| Imports | huggingface_hub |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides utility functions for interacting with the Hugging Face Hub API to retrieve model information and list models.

**Mechanism:** Uses a module-level singleton `_HFAPI` (lazily initialized `HfApi` instance) for efficient API access. The `get_model_info()` function retrieves detailed information about a specific model by ID, with configurable properties to fetch (defaults to safetensors and lastModified for minimal data). The `list_models()` function searches and lists models with filtering by author (defaults to "unsloth"), search query, sort order (defaults to downloads), and limit. The `formatted_int()` helper formats large numbers with K/M/B suffixes for display. Constants `POPULARITY_PROPERTIES` list available popularity metrics (downloads, downloadsAllTime, trendingScore, likes) and `THOUSAND`/`MILLION`/`BILLION` define numeric thresholds.

**Significance:** Utility module for discovering and inspecting models on Hugging Face Hub. Useful for programmatically finding Unsloth-optimized models, checking model metadata, and building model selection interfaces.
