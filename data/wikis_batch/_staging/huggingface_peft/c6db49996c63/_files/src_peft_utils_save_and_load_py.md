# File: `src/peft/utils/save_and_load.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 724 |
| Functions | `has_valid_embedding_base_layer`, `get_embedding_layer_name`, `get_peft_model_state_dict`, `set_peft_model_state_dict`, `torch_load`, `load_peft_weights` |
| Imports | __future__, constants, huggingface_hub, os, other, peft, peft_types, platform, re, safetensors, ... +4 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Manages serialization and deserialization of PEFT adapter weights and state dictionaries.

**Mechanism:** Extracts adapter-specific parameters from full model state dicts, handles embedding layer special cases, loads weights from disk (safetensors/pickle format), sets adapter state into models, and manages HuggingFace Hub integration for downloading adapter checkpoints.

**Significance:** Critical I/O layer that enables PEFT model persistence, sharing via HuggingFace Hub, and checkpoint management, ensuring adapters can be saved/loaded independently from base models while handling various edge cases like tied embeddings and quantized models.
