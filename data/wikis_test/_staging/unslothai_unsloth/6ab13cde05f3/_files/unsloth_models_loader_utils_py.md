# File: `unsloth/models/loader_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 423 |
| Functions | `is_distributed`, `prepare_device_map`, `get_model_name` |
| Imports | device_type, gc, importlib, mapper, os, packaging, re, tempfile, torch, transformers, ... +3 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Model loading and distributed training setup

**Mechanism:** Infers distributed ranks, prepares device maps for multi-GPU

**Significance:** Enables seamless multi-GPU training
