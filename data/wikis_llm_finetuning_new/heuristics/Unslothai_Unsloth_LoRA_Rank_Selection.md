# Heuristic: LoRA Rank and Alpha Selection

## Category
Training/Configuration

## Summary
Selecting appropriate LoRA rank (r) and alpha values is crucial for balancing model expressiveness, memory efficiency, and training stability. This heuristic provides guidance based on task complexity and resource constraints.

## The Decision Framework

### LoRA Rank Selection Matrix

| Task Complexity | Recommended Rank | Memory Impact | Quality |
|----------------|------------------|---------------|---------|
| Simple adaptation | 8 | Lowest | Good |
| General fine-tuning | 16-32 | Low-Medium | Better |
| Complex tasks | 64 | Medium | Best |
| Full capability | 128+ | High | Maximum |

### Alpha to Rank Ratio

The common heuristic is `alpha = 2 * rank`:

| Rank | Alpha | Scaling Factor |
|------|-------|----------------|
| 8 | 16 | 2.0 |
| 16 | 32 | 2.0 |
| 32 | 64 | 2.0 |
| 64 | 128 | 2.0 |

## Implementation Evidence

From `unsloth/save.py:84-92`, the target modules for LoRA:

```python
LLAMA_WEIGHTS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
```

### Trainable Parameters Calculation

```
trainable_params = 2 * rank * hidden_size * num_target_modules * num_layers
```

For a 7B model with rank=16:
- Hidden size: 4096
- Target modules: 7 (q, k, v, o, gate, up, down)
- Layers: 32
- Trainable: 2 * 16 * 4096 * 7 * 32 = ~29M params (0.4% of model)

## Tribal Knowledge

### Task-Based Recommendations

| Use Case | Rank | Dropout | Target Modules |
|----------|------|---------|----------------|
| Chat/instruction | 16 | 0.05 | All attention + MLP |
| Domain adaptation | 32 | 0.1 | All attention + MLP |
| Style transfer | 8 | 0.05 | Attention only |
| Code generation | 64 | 0.1 | All layers |

### Common Pitfalls

1. **Rank too low**: Model can't learn task adequately
2. **Rank too high**: Overfitting, memory waste
3. **Alpha too low**: Slow learning, underfitting
4. **Alpha too high**: Training instability

## Decision Tree

```
Start
  │
  ├─ Task complexity?
  │   ├─ Simple (style, format) → rank=8, alpha=16
  │   ├─ Medium (instruction) → rank=16, alpha=32
  │   └─ Complex (reasoning) → rank=32-64, alpha=64-128
  │
  ├─ Dataset size?
  │   ├─ Small (<1k) → Lower rank to prevent overfitting
  │   └─ Large (>10k) → Can use higher rank
  │
  └─ VRAM constraints?
      ├─ Limited → rank=8-16
      └─ Ample → rank=32-64
```

## Configuration Examples

### Memory-Efficient Setup
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

### High-Quality Setup
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
```

### Embedding Training
From `trainer.py:139-179`:

```python
def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr=5e-5,
):
    # Separate learning rate for embeddings
    if name.endswith("modules_to_save.default.weight"):
        print(f"Unsloth: Setting lr = {embedding_lr:.2e} for {partial_name}.")
        param_groups["embeddings"][name] = param
```

## Best Practices

### DO
- Start with rank=16 as baseline
- Use alpha = 2 * rank as starting point
- Target all projection layers for best results
- Consider separate embedding learning rate

### DON'T
- Use rank > 128 without strong justification
- Forget dropout for regularization
- Apply LoRA to LayerNorm layers
- Use same rank for all model sizes

## Source Evidence

- Target Modules: `unsloth/save.py:84-92`
- Embedding LR: `unsloth/trainer.py:139-179`
- PEFT Integration: `unsloth/save.py:26-28`

## Backlinks

[[used_by::Implementation:Unslothai_Unsloth_get_peft_model]]
[[used_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained]]

## Related

- [[Heuristic:Unslothai_Unsloth_Batch_Size_Selection]]
- [[Environment:Unslothai_Unsloth_PEFT]]
