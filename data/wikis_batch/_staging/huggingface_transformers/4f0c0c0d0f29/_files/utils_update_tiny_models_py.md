# File: `utils/update_tiny_models.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 171 |
| Functions | `get_all_model_names`, `get_tiny_model_names_from_repo`, `get_tiny_model_summary_from_hub` |
| Imports | argparse, create_dummy_models, huggingface_hub, json, multiprocessing, os, time, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Orchestrates batch creation and upload of tiny test models to hf-internal-testing Hub organization for pipeline testing.

**Mechanism:** Identifies all model classes from auto modeling mappings, checks which tiny models already exist in tiny_model_summary.json to skip, calls create_dummy_models.create_tiny_models() with multiprocessing support to generate missing models, and can also fetch metadata from existing Hub models to rebuild the summary JSON. Configured for CI workflows with hardcoded parameters (upload=True, no_check=True).

**Significance:** Test infrastructure maintenance tool that ensures every model architecture has a corresponding tiny model for fast integration testing. Keeps tiny_model_summary.json synchronized with Hub, enabling reliable pipeline tests across all models without requiring full-size model downloads. Critical for CI sustainability and test speed.
