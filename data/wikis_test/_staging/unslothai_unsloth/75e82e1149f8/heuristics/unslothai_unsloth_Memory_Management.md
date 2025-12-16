# Heuristic: Memory Management

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Discussion|GPU OOM Issues|https://github.com/unslothai/unsloth/issues]]
|-
! Domains
| [[domain::LLMs]], [[domain::Optimization]], [[domain::Infrastructure]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

## Overview

Memory optimization strategies for training large models with limited GPU VRAM.

### Description

Unsloth implements several memory optimization techniques that work together:
- 4-bit NF4 quantization via bitsandbytes
- Gradient checkpointing (standard and "unsloth" mode)
- Sequence packing / padding-free training
- Global buffer reuse for dequantization

### Usage

Apply these heuristics when:
- Training on consumer GPUs (8-24GB VRAM)
- Encountering CUDA OOM errors
- Trying to maximize batch size for throughput

## The Insight (Rule of Thumb)

### GPU Memory Estimation

| Model Size | Minimum VRAM | Recommended VRAM | Batch Size Guidance |
|------------|--------------|------------------|---------------------|
| 1-3B | 8GB | 16GB | batch_size=4-8 |
| 7-8B | 16GB | 24GB | batch_size=2-4 |
| 13B | 24GB | 40GB | batch_size=1-2 |
| 70B | 80GB+ | 2x 80GB | batch_size=1 |

### Maximum Memory Usage Parameter

* **Action:** Set `maximum_memory_usage=0.9` (default) in save functions
* **Range:** 0.1 to 0.95 (must be <= 0.95)
* **Trade-off:** Higher values risk OOM; lower values waste available memory

From `unsloth/save.py:300`:
```python
assert maximum_memory_usage > 0 and maximum_memory_usage <= 0.95
```

### Gradient Checkpointing Modes

* **`use_gradient_checkpointing=True`:** Standard HuggingFace implementation
* **`use_gradient_checkpointing="unsloth"`:** Optimized selective checkpointing
  - Reduces VRAM by ~30-50% compared to no checkpointing
  - ~10-20% slower training vs. no checkpointing
  - **Recommended for most users**

### Sample Packing

* **Action:** Enable `packing=True` in SFTConfig
* **Benefit:** Fills sequences to max_seq_length, reducing padding waste
* **Trade-off:** May not work with all models (gemma2, gpt_oss blocked)
* **Alternative:** `padding_free=True` for automatic padding-free batching

From `unsloth/trainer.py:56-59`:
```python
PADDING_FREE_BLOCKLIST = {
    "gemma2",  # Uses slow_attention_softcapping
    "gpt_oss",  # Uses Flex Attention
}
```

### Global Buffer Reuse

From `unsloth/kernels/utils.py:384-406`:
```python
if use_global_buffer:
    # Use same buffers for faster inference
    size = shape[0] * shape[1]
    global WEIGHT_BUFFERS
    global ABSMAX_BUFFERS
    # ... reuse existing buffers
```

## Reasoning

### Why 4-bit Quantization Works

The NF4 (4-bit NormalFloat) data type is optimal for normally-distributed weights:
- 4-bit weights use 1/4 the memory of float16
- Double quantization adds minimal overhead but further reduces memory
- Compute happens in float16/bfloat16 for accuracy

### Why Gradient Checkpointing

Deep transformers store massive activation tensors during forward pass:
- Activation memory scales with: batch_size × seq_length × hidden_dim × num_layers
- Checkpointing trades ~20% compute for ~50% memory reduction
- Unsloth's "unsloth" mode is selective, avoiding unnecessary recomputation

### Disk Spillover for Large Saves

From `unsloth/save.py:648-661`:
```python
else:
    # Save to Disk
    logger.warning_once("\nWe will save to Disk and not RAM now.")
    filename = os.path.join(temporary_location, f"{name}.pt")
    torch.save(W, filename, ...)
    state_dict[name] = torch.load(filename, map_location="cpu", mmap=True, ...)
```

## Related Pages

### Used By

* [[uses_heuristic::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[uses_heuristic::Implementation:unslothai_unsloth_UnslothTrainer]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[uses_heuristic::Principle:unslothai_unsloth_Gradient_Checkpointing]]
