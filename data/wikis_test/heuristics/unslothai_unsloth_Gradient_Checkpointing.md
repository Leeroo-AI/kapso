# Heuristic: unslothai_unsloth_Gradient_Checkpointing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|Gradient Checkpointing|https://arxiv.org/abs/1604.06174]]
|-
! Domains
| [[domain::Memory_Optimization]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

## Overview

Memory optimization via Unsloth's smart gradient checkpointing that reduces VRAM by ~30% with minimal speed impact.

### Description

Gradient checkpointing trades compute for memory by not storing all intermediate activations during the forward pass. Instead, it recomputes them during backpropagation. Unsloth implements a "smart" version that:

1. Selectively checkpoints only the most memory-intensive layers
2. Uses optimized recomputation patterns
3. Patches PyTorch's autograd for better efficiency

The default `use_gradient_checkpointing="unsloth"` mode provides ~30% VRAM savings with only ~5-10% slowdown.

### Usage

Use this heuristic when:
- Training larger models than your VRAM allows
- Getting CUDA OOM errors during training
- Maximizing batch size for better throughput

## The Insight (Rule of Thumb)

* **Action:** Set `use_gradient_checkpointing="unsloth"` (default in Unsloth)
* **Value Options:**
  - `"unsloth"` (recommended): Smart selective checkpointing, ~30% VRAM savings
  - `True`: Standard HuggingFace gradient checkpointing
  - `False`: Disable, maximum speed but highest memory
* **Trade-off:** ~30% VRAM reduction for ~5-10% training slowdown
* **Compatibility:** Works with all Transformer architectures; requires `use_cache=False` during training

### When to Use

```
VRAM tight but training works?  → "unsloth" (default)
Getting OOM errors?             → "unsloth" or True
Maximum training speed needed?  → False
Using mixed precision?          → "unsloth" (handles dtype correctly)
```

## Reasoning

Standard gradient checkpointing stores activations at layer boundaries and recomputes intermediate values. This adds ~33% compute but reduces memory by ~50%.

Unsloth's "smart" checkpointing:
1. Analyzes model architecture to find optimal checkpoint points
2. Only checkpoints layers with high activation memory
3. Patches `torch.autograd` for efficient recomputation
4. Handles dtype casting (float16/bfloat16/float32) correctly

The result: Better memory/speed tradeoff than vanilla gradient checkpointing.

## Code Evidence

From `unsloth/models/loader.py:135`:
```python
use_gradient_checkpointing = "unsloth",  # Default value
```

From `unsloth/models/loader.py:521-522`:
```python
if use_gradient_checkpointing == "unsloth":
    patch_unsloth_smart_gradient_checkpointing(dtype = dtype)
```

From `unsloth/models/llama.py:2642-2645`:
```python
if use_gradient_checkpointing == "unsloth":
    patch_unsloth_smart_gradient_checkpointing(
        dtype = model.get_input_embeddings().weight.dtype
    )
```

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[uses_heuristic::Principle:unslothai_unsloth_Environment_Setup]]
