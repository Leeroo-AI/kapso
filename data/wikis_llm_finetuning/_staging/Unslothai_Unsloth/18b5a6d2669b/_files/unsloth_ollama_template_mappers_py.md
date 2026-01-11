# File: `unsloth/ollama_template_mappers.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 2192 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Bidirectional mapping between model names and Ollama template identifiers

**Mechanism:** Defines OLLAMA_TEMPLATES dictionary with 50+ Modelfile templates keyed by template names (unsloth, zephyr, mistral, llama variants, gemma, phi, qwen, etc), OLLAMA_TEMPLATE_TO_MODEL_MAPPER providing template-to-model-list mappings, and MODEL_TO_OLLAMA_TEMPLATE_MAPPER (reverse index) built at module load time for efficient model-to-template lookups

**Significance:** Critical reference data structure enabling automatic Ollama Modelfile generation for saved models, supporting deployment of fine-tuned models to local Ollama instances without manual configuration
