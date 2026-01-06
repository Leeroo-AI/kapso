# File: `utils/create_dummy_models.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1479 |
| Functions | `get_processor_types_from_config_class`, `get_architectures_from_config_class`, `get_config_class_from_processor_class`, `build_processor`, `get_tiny_config`, `convert_tokenizer`, `convert_feature_extractor`, `convert_processors`, `... +13 more` |
| Imports | argparse, check_config_docstrings, collections, copy, datasets, get_test_info, huggingface_hub, inspect, json, multiprocessing, ... +6 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Generates tiny random model checkpoints for testing purposes. Creates miniature versions of transformer models with minimal parameters and vocabulary sizes suitable for fast CI/CD testing.

**Mechanism:** Uses model testers from test files to extract tiny configurations with reduced dimensions. Builds processors (tokenizers, feature extractors) by training new tokenizers on WikiText-2 dataset with target vocab size of 1024, then creates models with these tiny configs. Supports multiprocessing for batch creation and can upload results to HuggingFace Hub. Handles composite models (encoder-decoder, vision-encoder-decoder) by building component models separately then combining them.

**Significance:** Critical testing infrastructure that enables fast integration tests without requiring full-size models. Reduces CI time and resource consumption while maintaining architecture coverage. Used by the test suite to validate model implementations across hundreds of architectures.
