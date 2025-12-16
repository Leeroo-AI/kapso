# Heuristic: unslothai_unsloth_Memory_Optimization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|save.py|unsloth/save.py]]
* [[source::Doc|llama.py|unsloth/models/llama.py]]
* [[source::Doc|rl.py|unsloth/models/rl.py]]
|-
! Domains
| [[domain::Optimization]], [[domain::LLMs]], [[domain::Memory_Management]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

## Overview

Memory optimization techniques for training and inference, including gradient checkpointing, VRAM management, and efficient saving strategies.

### Description

Unsloth implements multiple memory optimization techniques to enable training large models on consumer hardware. Key strategies include:

- **4-bit quantization**: Reduces base model weights from 16 bytes to 0.5 bytes per parameter
- **Gradient checkpointing**: Trades compute for memory by recomputing activations
- **Triton kernels**: Fused operations reduce memory bandwidth requirements
- **Paged KV cache**: Efficient memory allocation during inference
- **Smart saving**: Chunks model saving to avoid OOM during export

### Usage

Apply these techniques when:
- Training models on GPUs with limited VRAM (8-24GB)
- Experiencing CUDA OOM errors during training
- Exporting large merged models
- Running inference with KV cache growth

## The Insight (Rule of Thumb)

### Training Memory

* **4-bit Loading**: Always use `load_in_4bit=True` for training:
  - 7B model: ~4GB VRAM (vs ~14GB for 16-bit)
  - 13B model: ~8GB VRAM (vs ~26GB for 16-bit)

* **Gradient Checkpointing**: Enabled by default via `use_gradient_checkpointing="unsloth"`:
  - Reduces memory by ~50% at cost of ~20% slower training
  - Use `gradient_checkpointing_kwargs={"use_reentrant": False}` for better compatibility

* **Batch Size Tuning**:
  - Start with `per_device_train_batch_size=1`
  - Increase `gradient_accumulation_steps` instead of batch size
  - Effective batch = batch_size * gradient_accumulation * num_gpus

### Saving Memory

From `save.py:300`:
```python
assert maximum_memory_usage > 0 and maximum_memory_usage <= 0.95
```

* **Action**: Set `maximum_memory_usage=0.9` to leave 10% headroom
* **For Colab/Kaggle**: Delete cached model to free disk space (handled automatically)
* **For slow PCs (<=2 CPUs)**: Disable safe_serialization for 10x faster saving

### Inference Memory

* **KV Cache Increment**: Default 512 tokens - adjust for long context:
  ```python
  KV_CACHE_INCREMENT = 512  # From llama.py:144
  ```

* **vLLM Memory**: For RL training with vLLM:
  - `gpu_memory_utilization=0.5` leaves room for training
  - Higher values may cause OOM when switching training/inference

### Environment-Specific

* **Colab**: Limited to ~15GB disk, auto-frees cached models
* **Kaggle**: Uses `/tmp` for temporary files to avoid quota issues
* **2-CPU machines**: Auto-switches to faster pickle serialization

## Reasoning

Memory is the primary bottleneck for local LLM training. Unsloth's 2x memory reduction comes from:

1. **Fused kernels**: Combine multiple operations to reduce intermediate tensors
2. **No weight offloading**: Keep everything on GPU but quantized
3. **Minimal optimizer states**: 4-bit with LoRA has tiny optimizer footprint

**Why gradient checkpointing works:**
Instead of storing O(layers) activation tensors, store only O(sqrt(layers)) checkpoints and recompute during backward pass.

**Memory formula for 4-bit LoRA:**
```
Total VRAM ≈ Model_params / 8 * 0.5GB (4-bit weights)
           + Batch * Seq * Hidden * 2 (activations)
           + LoRA_params * 4 * 2 (optimizer states)
           + Overhead (~1-2GB)
```

## Code Evidence

Memory management during saving from `save.py:536-584`:
```python
# Determine max RAM usage minus sharding
max_ram = psutil.virtual_memory().available
sharded_ram_usage = 5 * 1024 * 1024 * 1024  # 5GB default
if type(max_shard_size) is str:
    gb_found = re.match(r"([0-9]{1,})[\s]{0,}GB", max_shard_size, flags=re.IGNORECASE)
    if gb_found:
        sharded_ram_usage = int(gb_found.group(1)) * 1024 * 1024 * 1024

# Switch to faster saving for slow PCs
n_cpus = psutil.cpu_count(logical=False)
if n_cpus is None:
    n_cpus = 1
if safe_serialization and (n_cpus <= 2):
    logger.warning_once(
        f"Unsloth: You have {n_cpus} CPUs. Using `safe_serialization` is 10x slower.\n"
        f"We shall switch to Pytorch saving, which might take 3 minutes and not 30 minutes."
    )
    safe_serialization = False
    save_function = fast_save_pickle
```

Colab/Kaggle handling from `save.py:596-602`:
```python
if IS_KAGGLE_ENVIRONMENT or IS_COLAB_ENVIRONMENT:
    logger.warning_once(
        "Unsloth: Kaggle/Colab has limited disk space. We need to delete the downloaded\n"
        "model which will save 4-16GB of disk space, allowing you to save on Kaggle/Colab."
    )
    _free_cached_model(internal_model)
```

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[uses_heuristic::Implementation:unslothai_unsloth_save_pretrained_merged]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
