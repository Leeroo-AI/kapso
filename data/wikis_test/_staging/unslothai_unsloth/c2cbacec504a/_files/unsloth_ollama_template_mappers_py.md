# File: `unsloth/ollama_template_mappers.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 2192 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Maps model architectures to their corresponding Ollama template formats for compatibility with Ollama inference engine.

**Mechanism:** Defines OLLAMA_TEMPLATES dictionary with Ollama-specific prompt templates using Ollama's template syntax for different model types (unsloth, zephyr, ChatML, Mistral, Llama, Dolphin, etc.). Provides MODEL_TO_OLLAMA_TEMPLATE_MAPPER to lookup appropriate template by model name or type.

**Significance:** Enables seamless integration with Ollama local inference engine by ensuring models use correct prompt formatting. Allows users to run Unsloth-fine-tuned models locally with Ollama.
