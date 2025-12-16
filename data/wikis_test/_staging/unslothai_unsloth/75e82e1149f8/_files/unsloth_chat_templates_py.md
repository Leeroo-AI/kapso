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

**Purpose:** Comprehensive chat template management for conversation formatting across 50+ model families.

**Mechanism:**
- `get_chat_template()`: Returns Jinja2 template for specified model (llama-3, chatml, mistral, etc.)
- Supports mapping aliases (e.g., "llama-3.1" -> llama-3 template)
- `apply_chat_template()`: Applies template to conversation with proper special tokens
- `construct_chat_template()`: Builds template from components (system, user, assistant formats)
- `to_sharegpt()`: Converts HF-style messages to ShareGPT format
- `get_ollama_eos_tokens()`: Extracts EOS tokens for Ollama Modelfile generation
- `create_stopping_criteria()`: Builds stopping criteria for generation based on EOS tokens
- `StoppingCriteriaSub`: Custom stopping criteria class for multi-token EOS sequences
- Handles complex templates: tool calling, thinking tags, multi-turn with system prompts
- Template registry includes: Llama, Mistral, Phi, Qwen, Gemma, DeepSeek, Cohere, and many more

**Significance:** Critical for correct inference and training. Wrong chat templates cause degraded performance or complete failure. This centralizes template handling for all supported models.
