# File: `unsloth/chat_templates.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 3159 |
| Classes | `StoppingCriteriaSub` |
| Functions | `get_chat_template`, `remove_special_tokens`, `to_sharegpt`, `get_ollama_eos_tokens`, `construct_chat_template`, `test_construct_chat_template`, `apply_chat_template`, `create_stopping_criteria`, `... +2 more` |
| Imports | models, os, re, save, shutil, tokenizer_utils, torch, transformers, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Manages chat template definitions for different model architectures and provides utilities for template management, standardization, and validation across multiple chat formats.

**Mechanism:** Defines CHAT_TEMPLATES dictionary mapping model types to Jinja2 format templates with special tokens, EOS behavior, and optional system messages. Implements functions like get_chat_template() to retrieve appropriate templates, construct_chat_template() to build new ones, and test_chat_templates() to validate tokenizer/template compatibility. Handles conversion between ShareGPT and standard formats.

**Significance:** Critical for correct chat prompt formatting across different model families. Ensures models receive properly formatted input matching their training format, which is essential for accurate inference and fine-tuning.
