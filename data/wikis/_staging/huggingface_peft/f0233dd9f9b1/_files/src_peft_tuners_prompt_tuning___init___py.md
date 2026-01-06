# File: `src/peft/tuners/prompt_tuning/__init__.py`

**Category:** tuner module initialization

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | peft.utils.register_peft_method, .config, .model |
| Exports | `PromptEmbedding`, `PromptTuningConfig`, `PromptTuningInit` |

## Understanding

**Status:** Explored

**Purpose:** Initializes the Prompt Tuning PEFT method module by importing and exporting key components, and registering the method with PEFT's method registry.

**Mechanism:**
- Imports the core Prompt Tuning components:
  - `PromptTuningConfig`: Configuration class for Prompt Tuning
  - `PromptTuningInit`: Enum for initialization strategies (TEXT, SAMPLE_VOCAB, RANDOM)
  - `PromptEmbedding`: Model implementation
- Registers Prompt Tuning with PEFT framework using `register_peft_method()`:
  - Method name: "prompt_tuning"
  - Config class: PromptTuningConfig
  - Model class: PromptEmbedding
- Exports all three classes in `__all__` for public API

**Significance:** Critical module initialization file that integrates Prompt Tuning into the PEFT framework. The registration call enables PEFT to dynamically instantiate Prompt Tuning adapters when users specify `peft_type="PROMPT_TUNING"` in their configs. Prompt Tuning is the simplest prompt learning method, directly learning continuous embeddings that are prepended to input sequences.
