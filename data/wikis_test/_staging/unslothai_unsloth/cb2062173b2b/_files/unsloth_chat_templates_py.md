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

**Purpose:** Provides comprehensive chat template management for instruction-tuned models, supporting 30+ template formats (Llama, Mistral, Gemma, Qwen, ChatML, Vicuna, etc.) with corresponding Ollama conversions. Handles dataset formatting and standardization for conversational AI training.

**Mechanism:**
- **Template database**: `CHAT_TEMPLATES` dictionary stores Jinja2 templates for each model family (unsloth, zephyr, chatml, mistral, llama, llama-3, phi-3, gemma, qwen, etc.)
- **Template components**: Each entry contains (jinja2_template, eos_token_spec, is_custom_eos, ollama_template)
- **System messages**: `DEFAULT_SYSTEM_MESSAGE` dict provides default system prompts per template type
- **Template formats**: Handles special tokens (BOS/EOS), role markers, system prompts, multi-turn conversations
- **Ollama integration**: Parallel `OLLAMA_TEMPLATES` with Modelfile syntax for local deployment via Ollama
- **Template testing**: `test_chat_templates()` validates template correctness by comparing HuggingFace and GGUF tokenization
- **Dataset utilities**:
  - `standardize_data_formats()`: Converts various formats (ShareGPT, conversations, etc.) to standard format
  - `train_on_responses_only()`: Masks prompts so loss is computed only on assistant responses
  - `apply_chat_template()`: Applies template to conversations with proper special token handling
- **Special token handling**: Manages complex EOS scenarios (e.g., Gemma's `<end_of_turn>`, ChatML's `<|im_end|>`)
- **Model-specific variations**: Handles template differences between versions (Mistral v0.1/v0.2/v0.3, Gemma 1/2, Phi-3/3.5/4)

**Significance:** Chat templates are critical for instruction-tuned model quality. Incorrect templates cause models to hallucinate, ignore instructions, or produce malformed output. This module centralizes template knowledge, preventing common errors like missing BOS/EOS tokens, incorrect role markers, or broken multi-turn context. The Ollama integration enables seamless local deployment of fine-tuned models. Supporting 30+ templates makes Unsloth compatible with most popular open-source chat models.
