# Heuristic: Batch Size and Gradient Accumulation Selection

## Category
Training/Performance

## Summary
Selecting optimal batch size and gradient accumulation steps is critical for training efficiency. Unsloth provides guidance for balancing VRAM usage, training speed, and model convergence.

## The Decision Framework

### Batch Size Selection Matrix

| VRAM | Model Size | Recommended Per-Device Batch | Gradient Accum |
|------|------------|------------------------------|----------------|
| 8GB | 7B (4-bit) | 1-2 | 8-16 |
| 16GB | 7B (4-bit) | 4-8 | 4-8 |
| 24GB | 7B (4-bit) | 8-16 | 2-4 |
| 24GB | 13B (4-bit) | 2-4 | 8-16 |
| 40GB+ | 70B (4-bit) | 1-2 | 16-32 |

### Effective Batch Size Formula

```
effective_batch_size = per_device_batch * gradient_accumulation_steps * num_gpus
```

## Tribal Knowledge

### Recommended Effective Batch Sizes

| Task Type | Min Effective | Optimal | Max Useful |
|-----------|--------------|---------|------------|
| Instruction tuning | 8 | 32-64 | 128 |
| Continued pretraining | 64 | 256-512 | 2048 |
| Chat fine-tuning | 16 | 64-128 | 256 |

### Memory Estimation

From code analysis, memory usage scales with:
1. **Model parameters**: Base memory footprint
2. **Batch size**: Linear scaling
3. **Sequence length**: Quadratic for attention
4. **Gradient checkpointing**: ~40-60% reduction

## Implementation Evidence

From `unsloth/save.py:543-558`:

```python
# Determine max RAM usage minus sharding
max_ram = psutil.virtual_memory().available
sharded_ram_usage = 5 * 1024 * 1024 * 1024
if type(max_shard_size) is str:
    gb_found = re.match(
        r"([0-9]{1,})[\s]{0,}GB", max_shard_size, flags=re.IGNORECASE
    )
    mb_found = re.match(
        r"([0-9]{1,})[\s]{0,}MB", max_shard_size, flags=re.IGNORECASE
    )
```

From `trainer.py`, gradient accumulation is handled via TRL integration:

```python
# Unsloth gradient accumulation fix for transformers < 4.45.2
if Version(transformers_version) > Version("4.45.2"):
    def unsloth_train(trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)
else:
    # Custom gradient accumulation fix
    return _unsloth_train(trainer)
```

## Decision Tree

```
Start
  │
  ├─ Determine available VRAM
  │   └─ Run: torch.cuda.get_device_properties(0).total_memory
  │
  ├─ Calculate base memory for model
  │   └─ 4-bit: ~0.5GB per billion params
  │   └─ 16-bit: ~2GB per billion params
  │
  ├─ Estimate remaining VRAM for batch
  │   └─ remaining = total - model_size - 2GB_buffer
  │
  ├─ Set per_device_batch_size
  │   └─ Start low, increase until ~90% VRAM used
  │
  └─ Set gradient_accumulation_steps
      └─ To achieve target effective batch size
```

## Best Practices

### DO
- Start with small batch size and increase
- Monitor VRAM usage during first training steps
- Use gradient accumulation to achieve larger effective batches
- Match effective batch size to task requirements

### DON'T
- Set batch size without checking VRAM
- Ignore gradient accumulation (can cause OOM)
- Use very large batches for small datasets
- Forget to account for optimizer states

## Configuration Examples

### Memory-Constrained Setup
```python
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    # effective_batch_size = 32
)
```

### High-VRAM Setup
```python
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    # effective_batch_size = 32
)
```

## Source Evidence

- Memory Management: `unsloth/save.py:543-558`
- Trainer Integration: `unsloth/trainer.py`
- Gradient Accumulation Fix: `unsloth_zoo/training_utils.py`

## Backlinks

[[used_by::Implementation:Unslothai_Unsloth_SFTTrainer_train]]
[[used_by::Implementation:Unslothai_Unsloth_SFTConfig]]

## Related

- [[Heuristic:Unslothai_Unsloth_Gradient_Checkpointing]]
- [[Environment:Unslothai_Unsloth_CUDA_11]]
