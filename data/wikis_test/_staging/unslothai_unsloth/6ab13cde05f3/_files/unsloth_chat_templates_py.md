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

**Purpose:** Chat template formats for models

**Mechanism:** Jinja2-based chat templates and Ollama format mappings for many models

**Significance:** Essential for prompt formatting across model architectures
