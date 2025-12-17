# Implementation: adaption_prompt/config.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/adaption_prompt/config.py`
- **Size**: 88 lines
- **Description**: Configuration for LLaMA-Adapter style gated adaption prompts

## Overview

Adaption prompt is a lightweight adapter method that inserts trainable prompts into attention layers with gating mechanisms. Originally designed for LLaMA models, it enables efficient instruction following with minimal parameters.

## Core Components

### AdaptionPromptConfig

```python
@dataclass
class AdaptionPromptConfig(PeftConfig):
    target_modules: str = None          # Attention module name
    adapter_len: int = None             # Number of prompt tokens
    adapter_layers: int = None          # Number of top layers to adapt
```

### ModelTypeConfig

```python
ModelTypeConfig = namedtuple(
    "ModelTypeConfig",
    ["compute_query_states", "target_modules", "k_proj_layer", "v_proj_layer", "o_proj_layer"]
)
```

### Supported Models

**TRANSFORMERS_MODEL_CONFIG**:
- **llama/mistral**: Uses `llama_compute_query_states`, targets "self_attn"
- **gpt2**: Uses `gpt2_compute_query_states`, targets "attn"

## Key Features

- **Zero-init gating**: Starts with no interference
- **Top-layer only**: Adapts final layers for task-specific behavior
- **Minimal parameters**: Typically <1M parameters for 7B models

## Cross-References

- **Model**: `adaption_prompt/model.py`
- **Layer**: `adaption_prompt/layer.py`
- **Paper**: [LLaMA-Adapter](https://huggingface.co/papers/2303.16199)
