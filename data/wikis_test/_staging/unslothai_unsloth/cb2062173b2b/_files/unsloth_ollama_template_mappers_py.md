# File: `unsloth/ollama_template_mappers.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 2192 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides Ollama Modelfile templates for 50+ model families to enable local deployment of fine-tuned models. Maps HuggingFace model names to appropriate Ollama templates with correct prompt formats, stop tokens, and system messages.

**Mechanism:**
- **OLLAMA_TEMPLATES dictionary**: Maps template names (e.g., "mistral", "llama3", "qwen2.5") to Modelfile strings
- **Modelfile format**: Each template includes:
  - `FROM {__FILE_LOCATION__}`: Placeholder for GGUF file path
  - `TEMPLATE`: Ollama's template syntax using `{{ .System }}`, `{{ .Prompt }}`, `{{ .Response }}`
  - `PARAMETER stop`: Stop sequences (EOS tokens)
  - `PARAMETER temperature/min_p`: Default generation parameters
  - `SYSTEM`: Optional default system prompt
- **Model families covered**:
  - Mistral variants (v0.1, v0.2, v0.3, Nemo, Small, Large, Codestral, Devstral)
  - Meta models (Llama 2, Llama 3, Llama 3.1, Llama 3.2, Llama 3.3)
  - Google models (Gemma, Gemma 2)
  - Qwen series (1.5, 2, 2.5 including Coder/Math variants)
  - Microsoft Phi (3, 3.5, 4)
  - Other popular models (Vicuna, ChatML, Alpaca, Zephyr)
- **Template variations**: Handles version-specific differences (e.g., Mistral v0.1 uses simple [INST], v0.3 supports tools, Small uses [SYSTEM_PROMPT])
- **Tool calling support**: Advanced templates include tool syntax (e.g., `[TOOL_CALLS]`, `[AVAILABLE_TOOLS]`) for function calling models
- **System messages**: Defines default personas (e.g., Mistral Small's knowledge cutoff disclaimer)
- **Mappers**:
  - `OLLAMA_TEMPLATE_TO_MODEL_MAPPER`: Maps Ollama template names to HuggingFace model patterns
  - `MODEL_TO_OLLAMA_TEMPLATE_MAPPER`: Reverse mapping from model name patterns to template names
- **Special handling**: Multi-modal models, code completion (Codestral with PREFIX/MIDDLE/SUFFIX), chat variants

**Significance:** Ollama is the most popular tool for local LLM deployment. This module bridges HuggingFace training and Ollama deployment by generating correct Modelfiles automatically. Without proper templates, deployed models would ignore instructions, generate malformed output, or never stop generating. The comprehensive coverage (50+ templates) means users can fine-tune any popular model and immediately deploy locally. The template accuracy ensures production-quality results match training behavior. This enables the full cycle: train with Unsloth, export to GGUF, deploy with Ollama - all with one consistent interface.
