# File: `utils/check_config_docstrings.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 102 |
| Functions | `get_checkpoint_from_config_class`, `check_config_docstrings_have_checkpoints` |
| Imports | inspect, re, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Ensures all configuration class docstrings include a valid checkpoint link in the proper Markdown format for documentation consistency.

**Mechanism:** The script iterates through all configuration classes in `CONFIG_MAPPING`, retrieves their source code using `inspect.getsource()`, and searches for checkpoint references using a regex pattern that matches `[model-name](https://huggingface.co/model-name)` format. It verifies the checkpoint name and URL correspond correctly and maintains an ignore list for special cases (encoder-decoder models, synthetic configs like LlamaConfig without a single canonical checkpoint). If a config lacks a valid checkpoint link, the check fails.

**Significance:** Checkpoint links in config docstrings serve as canonical examples for users and are essential for documentation generation. They help users quickly find pretrained models for a given architecture and ensure documentation consistency across the library. This check is part of CI to prevent configurations from being added without proper documentation examples.
