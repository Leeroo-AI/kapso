# File: `src/peft/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 408 |
| Classes | `PeftConfigMixin`, `PeftConfig`, `PromptLearningConfig` |
| Imports | __future__, dataclasses, huggingface_hub, importlib, inspect, json, os, packaging, peft, transformers, ... +3 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines base configuration classes for all PEFT adapters with save/load functionality.

**Mechanism:** PeftConfigMixin provides save_pretrained/from_pretrained methods that serialize/deserialize configurations to/from JSON. Includes version tracking, forward compatibility handling (ignoring unknown kwargs from newer versions), and integration with Hugging Face Hub. PeftConfig adds base model information and inference mode. PromptLearningConfig extends for prompt-based methods.

**Significance:** Core infrastructure for PEFT configuration management. Ensures all PEFT methods have consistent configuration interfaces and can be saved/loaded reliably. The forward compatibility logic is crucial for maintaining library stability across versions. Serves as the base class for all tuner-specific configs.
