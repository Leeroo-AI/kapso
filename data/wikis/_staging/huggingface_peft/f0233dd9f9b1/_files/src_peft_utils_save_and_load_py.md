# File: `src/peft/utils/save_and_load.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 724 |
| Functions | `has_valid_embedding_base_layer`, `get_embedding_layer_name`, `get_peft_model_state_dict`, `set_peft_model_state_dict`, `torch_load`, `load_peft_weights` |
| Imports | __future__, constants, huggingface_hub, os, other, peft, peft_types, platform, re, safetensors, ... +4 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Handles serialization and deserialization of PEFT adapter weights, supporting multiple PEFT types (LoRA, prompt learning, etc.) with adapter name management, state dict remapping, and compatibility checks.

**Mechanism:** get_peft_model_state_dict() extracts adapter-specific weights from full model state, removing adapter names for portability. set_peft_model_state_dict() loads weights back, reinserting adapter names. load_peft_weights() downloads/loads from Hub or local files (safetensors/pickle). Handles special cases: DoRA magnitude vectors, VBLoRA topk compression, SHIRA/VERA indices, embedding layers, and auxiliary training wrappers.

**Significance:** Critical I/O component enabling adapter sharing and reuse. Supports save_embedding_layers="auto" to handle resized vocabularies, ignore_mismatched_sizes for flexible loading, and low_cpu_mem_usage for large models. Manages complex state dict transformations across PEFT methods, FSDP, and quantization backends.
