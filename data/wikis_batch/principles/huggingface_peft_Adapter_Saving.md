# Principle: Adapter Saving

> The conceptual step of persisting trained adapter weights for later use or sharing.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Persistence |
| **Workflow** | [[huggingface_peft_LoRA_Finetuning]] |
| **Step Number** | 5 |
| **Implementation** | [[huggingface_peft_save_pretrained]] |

---

## Concept

Adapter Saving is the final step in the training workflow where trained adapter weights are persisted to disk. This step leverages one of PEFT's key advantages: **only the adapter weights need to be saved**, not the entire base model.

Key aspects:
1. **Lightweight storage** - Adapters are typically 0.1-1% of base model size
2. **Easy sharing** - Upload to HuggingFace Hub for community use
3. **Version control friendly** - Small files can be tracked in git
4. **Multi-adapter management** - Save different adapters for different tasks

---

## Why This Matters

### Storage Efficiency

| Scenario | Full Fine-tune | PEFT Adapter |
|----------|---------------|--------------|
| 7B model checkpoint | ~14 GB | ~10-50 MB |
| 100 task variants | 1.4 TB | 1-5 GB |
| Cloud storage cost | $$$ | $ |

### Deployment Flexibility

Saved adapters enable:
- **Hot-swapping**: Change model behavior without reloading base model
- **A/B testing**: Compare adapter versions in production
- **Task routing**: Load different adapters per request
- **Iterative improvement**: Save checkpoints during training

---

## What Gets Saved

### Adapter Files

```
my-adapter/
├── adapter_config.json    # Configuration metadata
└── adapter_model.safetensors  # Trained weights
```

### Configuration Contents

The `adapter_config.json` captures all settings needed to reconstruct the adapter:

```json
{
  "base_model_name_or_path": "meta-llama/Llama-2-7b-hf",
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": 16,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
  "lora_dropout": 0.05,
  "bias": "none"
}
```

---

## Saving Patterns

### Pattern 1: Training Checkpoint

Save periodically during training:

```python
for epoch in range(num_epochs):
    train_one_epoch(model, dataloader)
    model.save_pretrained(f"./checkpoints/epoch-{epoch}")
```

### Pattern 2: Best Model

Save based on validation performance:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    model.save_pretrained("./best-adapter")
```

### Pattern 3: Multi-Adapter Save

Save specific adapters from a multi-adapter model:

```python
# Save only the task-specific adapter
model.save_pretrained(
    "./task-a-adapter",
    selected_adapters=["task_a"]
)
```

---

## Sharing Adapters

### HuggingFace Hub Upload

```python
# Direct push to Hub
model.push_to_hub(
    "username/my-lora-adapter",
    private=False,  # Make public
    token="hf_..."
)
```

### Manual Upload

```bash
# Using huggingface-cli
huggingface-cli upload username/my-adapter ./my-adapter
```

---

## Relationship to Workflow

```
[Load Base Model]
       |
       v
[LoRA Configuration]
       |
       v
[PEFT Application]
       |
       v
[Train the Adapter]
       |
       v
[Adapter Saving] <-- This Principle
```

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Use safetensors format | Faster, safer loading |
| Include base model info | Ensures reproducibility |
| Version your adapters | Track improvements |
| Document training details | Help users understand use case |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Saving full model | Wasted storage, slow | Use `save_pretrained` not torch.save |
| Missing config | Cannot load adapter | Always use PEFT save methods |
| Wrong base model | Adapter won't load | Document exact base model version |

---

## Implementation

Adapter saving is performed using `PeftModel.save_pretrained`:

[[implemented_by::Implementation:huggingface_peft_save_pretrained]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:LoRA_Finetuning]]
