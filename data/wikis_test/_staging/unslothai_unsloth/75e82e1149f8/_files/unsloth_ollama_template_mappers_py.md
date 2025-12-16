# File: `unsloth/ollama_template_mappers.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 2192 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive Ollama Modelfile templates for 40+ model families enabling local deployment via Ollama.

**Mechanism:**
- `OLLAMA_TEMPLATES`: Dictionary mapping template names to Ollama Modelfile strings
- Each template defines: FROM (model location), TEMPLATE (Go template for chat formatting), PARAMETER (stop tokens, temperature, etc.), SYSTEM (default system prompt)
- Templates use Go template syntax with `.System`, `.Prompt`, `.Response` placeholders
- `{__FILE_LOCATION__}` and `{__EOS_TOKEN__}` are replaced at export time
- Supports complex features: tool calling, multi-turn conversations, thinking tags
- Model families covered: Llama (2/3/3.1/3.2/3.3), Mistral (v0.1-v0.3, small, large), Qwen (1.5/2/2.5/3), Gemma (1/2/3), Phi (1.5/2/3/4), DeepSeek (V2/V3/R1), and many more
- `MODEL_TO_OLLAMA_TEMPLATE_MAPPER`: Maps HuggingFace model names to template names
- `OLLAMA_TEMPLATE_TO_MODEL_MAPPER`: Reverse mapping for template lookup

**Significance:** Enables seamless GGUF → Ollama deployment. Users can export models directly to Ollama-compatible format with correct chat templates, making local inference straightforward.
