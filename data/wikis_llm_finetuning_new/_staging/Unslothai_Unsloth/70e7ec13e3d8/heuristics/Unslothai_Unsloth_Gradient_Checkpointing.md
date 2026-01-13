# Heuristic: Gradient Checkpointing Configuration

## Category
Training/Memory Optimization

## Summary
Unsloth implements intelligent gradient checkpointing strategies to balance memory usage and training speed, with automatic configuration based on model size and available VRAM.

## The Decision Framework

### When to Enable Gradient Checkpointing

| Scenario | Recommendation | Rationale |
|----------|---------------|-----------|
| VRAM < 16GB | Enable | Required to fit model |
| VRAM 16-24GB | Enable for 7B+ models | Balance memory/speed |
| VRAM 24GB+ | Optional | Speed vs memory tradeoff |
| Long sequences (4k+) | Enable | Sequence length increases memory |

### Memory Savings vs Speed Tradeoff

| Checkpointing | Memory Reduction | Speed Impact |
|--------------|------------------|--------------|
| Disabled | Baseline | Fastest |
| Enabled | ~40-60% | ~20-30% slower |
| Aggressive | ~60-80% | ~40-50% slower |

## Implementation Evidence

From `unsloth/models/_utils.py`, gradient checkpointing is controlled via model configuration:

```python
# Gradient checkpointing setup
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
```

### Unsloth Optimization

Unsloth provides optimized gradient checkpointing that:
1. Reduces memory overhead compared to standard HF implementation
2. Maintains training stability
3. Works with Flash Attention

## Tribal Knowledge

### Best Practices

1. **Start with checkpointing enabled** for models >= 7B parameters
2. **Disable for small models** (< 3B) if VRAM allows
3. **Combine with 4-bit quantization** for maximum memory efficiency
4. **Monitor VRAM usage** during first training steps

### Common Pitfalls

1. **Over-aggressive checkpointing**: Can slow training significantly
2. **Forgetting to enable for fine-tuning**: Large models may OOM
3. **Incompatibility with some attention implementations**

## Decision Tree

```
Start
  │
  ├─ Model size > 7B?
  │   ├─ Yes → Enable gradient checkpointing
  │   └─ No → Check VRAM
  │           ├─ VRAM < 16GB → Enable
  │           └─ VRAM >= 16GB → Optional (speed preference)
  │
  ├─ Sequence length > 4096?
  │   └─ Yes → Enable gradient checkpointing
  │
  └─ Using 4-bit quantization?
      └─ Yes → Usually optional (already memory efficient)
```

## Configuration Examples

### Memory-Constrained (8-16GB VRAM)
```python
model = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=True,
    # Gradient checkpointing handled automatically
)
```

### Speed-Optimized (24GB+ VRAM)
```python
model = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=False,  # Full precision
    # May disable checkpointing for speed
)
```

## Source Evidence

- Memory management: `unsloth/models/_utils.py`
- Training integration: `unsloth/trainer.py`

## Backlinks

[[used_by::Implementation:Unslothai_Unsloth_get_peft_model]]
[[used_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained]]

## Related

- [[Heuristic:Unslothai_Unsloth_Batch_Size_Selection]]
- [[Environment:Unslothai_Unsloth_CUDA_11]]
