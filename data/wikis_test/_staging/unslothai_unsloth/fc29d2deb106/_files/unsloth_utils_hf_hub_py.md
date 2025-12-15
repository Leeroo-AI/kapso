# File: `unsloth/utils/hf_hub.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 78 |
| Functions | `formatted_int`, `get_model_info`, `list_models` |
| Imports | huggingface_hub |

## Understanding

**Status:** ✅ Explored

**Purpose:** HuggingFace Hub interaction utilities

**Mechanism:** Wraps HuggingFace Hub API through singleton HfApi instance, provides get_model_info() for retrieving metadata about specific models with configurable property expansion, list_models() for querying models by author/search/sort criteria, and formatted_int() for human-readable metric display (K/M/B suffixes).

**Significance:** Simplifies interaction with HuggingFace Hub for model discovery and validation, enabling Unsloth to verify model availability, retrieve metadata, and list available models without directly managing API connections or error handling.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
