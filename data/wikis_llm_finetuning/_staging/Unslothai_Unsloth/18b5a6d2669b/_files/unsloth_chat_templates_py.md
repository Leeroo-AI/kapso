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

**Purpose:** Chat template management and dataset formatting for instruction tuning

**Mechanism:** Defines CHAT_TEMPLATES dictionary with Jinja2 templates, EOS token configs, and Ollama modelfiles for 20+ model families (Llama 2/3/3.1/3.2/3.3, Mistral, Gemma 1/2/3/3n, Phi 3/3.5/4, Qwen 2.5/3, Vicuna, Alpaca, etc); provides get_chat_template(), construct_chat_template() for tokenizer setup, to_sharegpt() for ShareGPT conversion, and apply_chat_template() for dataset preprocessing with system message injection

**Significance:** Core data preprocessing component that standardizes instruction-tuned dataset formats across different model architectures, enables one-command dataset transformation to ShareGPT format, and ensures correct EOS token handling for chat-based fine-tuning
