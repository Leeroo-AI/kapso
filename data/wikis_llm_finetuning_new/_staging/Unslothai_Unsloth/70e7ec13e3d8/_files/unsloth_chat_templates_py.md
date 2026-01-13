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

**Purpose:** Manages chat/conversation templates for various LLM architectures (Llama, Qwen, Gemma, Phi, Mistral, etc.), enabling proper formatting of multi-turn conversations for fine-tuning and inference.

**Mechanism:** Provides `get_chat_template()` to retrieve model-specific Jinja2 templates, `apply_chat_template()` to format conversations, and `create_stopping_criteria()` for generation. Contains the `StoppingCriteriaSub` class for custom EOS handling. Supports 40+ template variants including alpaca, chatml, vicuna, zephyr, and model-specific formats. Integrates with Ollama templates for export compatibility.

**Significance:** Core component - essential for instruction fine-tuning workflows. Proper chat templating ensures training data matches the model's expected format, critical for fine-tuning quality.
