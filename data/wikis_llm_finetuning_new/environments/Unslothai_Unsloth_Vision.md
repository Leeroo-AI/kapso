# Environment: Vision Language Model Support

## Category
Software/Models

## Summary
Unsloth provides specialized support for Vision Language Models (VLMs), requiring PyTorch 2.4.0+ and specific handling for multimodal inputs, with automatic detection and configuration.

## Requirements

### Software Requirements
| Package | Version Constraint | Evidence |
|---------|-------------------|----------|
| PyTorch | >= 2.4.0 | Required for VLM support |
| transformers | >= 4.45.0 | VLM architecture support |
| Flash Attention | >= 2.6.3 | For efficient attention in vision models |

### Hardware Requirements
| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU VRAM | 16GB | 24GB+ | VLMs require more memory |
| CUDA Compute | 8.0+ | 8.9+ | For Flash Attention support |

## VLM Detection Logic

From `unsloth/models/_utils.py`:

```python
# Check if VLM
architectures = getattr(model_config, "architectures", None)
if architectures is None:
    architectures = []
is_vlm = any(
    x.endswith("ForConditionalGeneration") for x in architectures
)
is_vlm = is_vlm or hasattr(model_config, "vision_config")
```

From `unsloth/save.py:1853-1862`:
```python
is_vlm = False
if hasattr(self, "config") and hasattr(self.config, "architectures"):
    is_vlm = any(
        x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
        for x in self.config.architectures
    )
    is_vlm = is_vlm or hasattr(self.config, "vision_config")
```

## Supported VLM Architectures

| Architecture Pattern | Model Types |
|---------------------|-------------|
| `*ForConditionalGeneration` | LLaVA, Qwen-VL, etc. |
| `*ForVisionText2Text` | Vision-text models |
| Models with `vision_config` | Any model with vision encoder |

## Special Handling

### Data Collator
From `unsloth/trainer.py`:
```python
from unsloth_zoo.vision_utils import (
    UnslothVisionDataCollator,
)
```

### Packing Restrictions
VLMs cannot use sample packing or padding-free training:

From `trainer.py:318-326`:
```python
blocked = (
    (data_collator is not None)
    or isinstance(processing_class, ProcessorMixin)
    or is_vlm
    or is_unsupported_model
    or (os.environ.get("UNSLOTH_RETURN_LOGITS", "0") == "1")
)
```

### GGUF Export
VLMs require special mmproj file handling for GGUF export:

From `save.py:2027-2032`:
```python
if is_vlm_update:
    print(f"Unsloth: example usage for Multimodal LLMs: llama-mtmd-cli -m {all_file_locations[0]} --mmproj {all_file_locations[-1]}")
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `UNSLOTH_RETURN_LOGITS` | Force logit return (disables VLM optimizations) | `0` |

## Source Evidence

- VLM Detection: `unsloth/models/_utils.py`
- Trainer Handling: `unsloth/trainer.py:305-326`
- GGUF Export: `unsloth/save.py:1853-1862`
- Vision Utilities: `unsloth_zoo/vision_utils.py`

## Backlinks

[[required_by::Implementation:Unslothai_Unsloth_FastVisionModel_from_pretrained]]
[[required_by::Implementation:Unslothai_Unsloth_get_peft_model_vision]]
[[required_by::Implementation:Unslothai_Unsloth_UnslothVisionDataCollator]]

## Related

- [[Environment:Unslothai_Unsloth_CUDA_11]]
- [[Environment:Unslothai_Unsloth_TRL]]
