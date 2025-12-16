# Heuristic: unslothai_unsloth_Memory_Management

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Memory_Optimization]], [[domain::Training]], [[domain::Model_Export]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

## Overview

Techniques for managing GPU memory during training and model merging, including maximum_memory_usage tuning.

### Description

Unsloth provides several memory management strategies:

1. **`maximum_memory_usage`**: Controls GPU memory threshold during model merging (default 0.9 = 90%)
2. **Layer-by-layer processing**: Dequantizes and processes one layer at a time
3. **Cache clearing**: Aggressive `torch.cuda.empty_cache()` and `gc.collect()` calls
4. **CPU offloading**: Offloads original embeddings to CPU during training

### Usage

Use this heuristic when:
- Merging models causes OOM errors
- Training with embeddings causes memory issues
- Deploying on memory-constrained GPUs

## The Insight (Rule of Thumb)

### Memory Usage Parameter

* **Action:** Adjust `maximum_memory_usage` in save functions
* **Default Value:** 0.9 (90% of available VRAM)
* **Range:** 0.5 to 0.95
* **Trade-off:** Lower values = safer but slower merging

```python
model.save_pretrained_merged(
    save_directory,
    maximum_memory_usage=0.85,  # Reduce if OOM during merge
)
```

### Embedding Training Memory

When training `embed_tokens` or `lm_head`:
- Original modules offloaded to CPU
- Trainable modules stay on GPU in float32 (Tesla T4) or native dtype (others)
- Saves VRAM for large vocabulary models

### Memory Cleanup Pattern

```python
for _ in range(3):
    torch.cuda.empty_cache()
    gc.collect()
```

Used before and after major operations (merging, GGUF conversion).

## Reasoning

4-bit quantized models require dequantization to 16-bit for merging LoRA weights. For a 7B model:
- 4-bit weights: ~3.5GB
- 16-bit weights: ~14GB

Processing layer-by-layer with controlled memory threshold prevents OOM while allowing larger models to be processed.

The triple gc.collect() pattern ensures Python's garbage collector runs multiple generations, releasing cyclic references.

## Code Evidence

From `unsloth/save.py:251-253`:
```python
temporary_location: str = "_unsloth_temporary_saved_buffers",
maximum_memory_usage: float = 0.9,
# ...
assert maximum_memory_usage > 0 and maximum_memory_usage <= 0.95
```

From `unsloth/save.py:302-305`:
```python
# Clean memory up first
for _ in range(3):
    torch.cuda.empty_cache()
    gc.collect()
```

From `unsloth/models/llama.py:2714-2730`:
```python
if new_dtype == torch.float16:
    # See https://github.com/unslothai/unsloth/pull/1200
    # Tesla T4 must use float32 and not float16
    new_dtype = torch.float32

model.get_input_embeddings().modules_to_save.default.to(
    device = DEVICE_TYPE_TORCH, dtype = new_dtype, non_blocking = True
)
# [TODO] Move old embed_tokens to CPU - should be disk!
model.get_input_embeddings().original_module.to(
    device = "cpu", non_blocking = True
)
```

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_save_pretrained_merged]]
* [[uses_heuristic::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
