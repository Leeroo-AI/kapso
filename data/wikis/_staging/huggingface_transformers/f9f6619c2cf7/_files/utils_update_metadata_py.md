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

**Purpose:** Maintains the `huggingface/transformers-metadata` Hub dataset with current information about model support, pipelines, and auto classes.

**Mechanism:** Generates two datasets: (1) frameworks table showing which models support PyTorch and their appropriate processor classes (AutoTokenizer, AutoProcessor, etc.) by introspecting the transformers module, and (2) pipeline tags table mapping model classes to their pipeline tasks and auto classes using the `PIPELINE_TAGS_AND_AUTO_MODELS` constant. Downloads existing metadata from Hub, compares for changes, and uploads updated JSON files only if content differs. Also provides a check-only mode to validate all pipelines are defined.

**Significance:** Essential infrastructure for the Hugging Face Hub ecosystem that enables accurate model card generation, pipeline task inference, and auto-class recommendations. This metadata powers Hub features that help users discover the right models and understand how to use them. The automated updates ensure the metadata stays synchronized with the rapidly evolving transformers library.
