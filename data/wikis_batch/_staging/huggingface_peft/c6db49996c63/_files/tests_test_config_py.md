# File: `tests/test_config.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 599 |
| Classes | `TestPeftConfig` |
| Imports | copy, json, os, peft, pickle, pytest, tempfile, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for all PEFT configuration classes.

**Mechanism:** Contains `TestPeftConfig` class with comprehensive parametrized tests covering 25+ config types (AdaLora, BOFT, LoRA, etc.). Tests include: required methods (to_dict, save_pretrained, from_pretrained), task type validation, from_peft_type loading, cache_dir handling, runtime_config removal, config copying/pickling, adapter-specific validation (IA3 feedforward subsets, regex with layer indexing), AdaLora timing bounds, forward compatibility with unknown keys, PEFT version tracking, and dev version commit hash recording.

**Significance:** Ensures all configuration classes follow consistent patterns, handle serialization correctly, validate parameters properly, and maintain backward/forward compatibility across PEFT versions.
