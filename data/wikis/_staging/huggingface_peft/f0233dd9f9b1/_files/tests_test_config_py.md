# File: `tests/test_config.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 599 |
| Classes | `TestPeftConfig` |
| Imports | copy, json, os, peft, pickle, pytest, tempfile, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for PEFT configuration classes

**Mechanism:** Tests all PEFT config classes (LoraConfig, AdaLoraConfig, BOFTConfig, etc.) for serialization (to_dict, from_pretrained, save_pretrained), validation, pickling, deprecation warnings, and config-specific attributes across 20+ config types

**Significance:** Test coverage for configuration infrastructure across all PEFT methods
