# File: `unsloth/ollama_template_mappers.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 2192 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines Ollama Modelfile templates and bidirectional mappings between model architectures and their corresponding Ollama template names for GGUF export.

**Mechanism:** Contains `OLLAMA_TEMPLATES` dictionary with complete Modelfile templates for 40+ model families (Mistral, Llama, Qwen, Gemma, Phi, Granite, Yi, Starling, etc.). `OLLAMA_TEMPLATE_TO_MODEL_MAPPER` maps template names to model identifiers, while `MODEL_TO_OLLAMA_TEMPLATE_MAPPER` provides reverse lookup. Templates include system prompts, stop tokens, and parameter configurations specific to each model architecture.

**Significance:** Utility - enables seamless export of fine-tuned models to Ollama format. Critical for the save/export pipeline when users want to deploy models locally using Ollama.
