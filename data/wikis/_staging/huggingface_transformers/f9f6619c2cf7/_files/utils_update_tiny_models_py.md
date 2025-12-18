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

**Purpose:** Automates creation and uploading of tiny model versions for all model architectures to the `hf-internal-testing` organization on Hugging Face Hub.

**Mechanism:** Wraps `create_dummy_models.py` with pre-configured arguments to process all model types, skipping those already present in `tests/utils/tiny_model_summary.json`. Uses multiprocessing with spawn method to parallelize model creation. Can also fetch and generate a summary of existing tiny models from Hub by attempting to load tokenizers, processors, and models for each repository. The tiny models are minimal versions used for fast testing.

**Significance:** Critical testing infrastructure that maintains a comprehensive collection of lightweight model checkpoints for CI pipelines. These tiny models (with minimal layers/parameters) allow thorough testing of all model architectures without the memory and time costs of full-sized models. The automated updates ensure new model architectures get tiny versions immediately, keeping test coverage complete and CI fast.
