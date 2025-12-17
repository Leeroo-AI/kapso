# File: `utils/update_metadata.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 350 |
| Functions | `camel_case_split`, `get_frameworks_table`, `update_pipeline_and_auto_class_table`, `update_metadata`, `check_pipeline_tags` |
| Imports | argparse, collections, datasets, huggingface_hub, os, pandas, re, tempfile, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Maintains the transformers-metadata Hub dataset with current framework support and pipeline mappings for all model types.

**Mechanism:** Introspects transformers module to build a frameworks table (PyTorch support) and processor mapping (AutoTokenizer/AutoProcessor/etc.) for each model type, extracts pipeline tags and auto-class associations from MODEL_FOR_*_MAPPING_NAMES constants, merges with existing Hub metadata to preserve historical entries, and uploads updated JSON files only if changes detected. Includes validation mode to ensure new pipelines are properly registered.

**Significance:** Critical integration that powers the Hugging Face Hub's model card generation and pipeline task inference. Ensures model pages display correct framework badges and pipeline tags, enabling users to discover models by task and understand compatibility. Used by GitHub Actions to keep Hub metadata synchronized with library changes.
