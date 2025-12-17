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

**Purpose:** Generates tiny random model checkpoints for fast CI testing by creating miniaturized versions of all model architectures.

**Mechanism:** Uses model tester classes to extract tiny configurations (small hidden sizes, few layers, etc.). Builds processors by finding checkpoints via docstrings, retraining tokenizers with 1024 vocab size on WikiText-2, and adjusting feature extractors to tiny image sizes. Creates models from tiny configs, handles special cases like composite encoder-decoder models, and uploads to HuggingFace Hub under `hf-internal-testing/tiny-random-{ModelName}`. Supports multiprocessing for faster generation and includes extensive error handling with detailed reports.

**Significance:** Essential testing infrastructure that enables fast CI pipeline execution by providing lightweight model checkpoints (KB instead of GB) while maintaining architectural compatibility for comprehensive integration testing.
