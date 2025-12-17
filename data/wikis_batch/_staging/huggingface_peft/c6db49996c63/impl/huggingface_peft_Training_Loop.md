# Implementation: Training_Loop

> Wrapper Documentation for training PEFT adapters using HuggingFace Trainer or custom loops.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | Wrapper Doc |
| **Source** | External (`transformers` library) |
| **Class** | `Trainer` or custom PyTorch loop |
| **Paired Principle** | [[huggingface_peft_Adapter_Training]] |
| **Parent Workflow** | [[huggingface_peft_LoRA_Finetuning]] |

---

## Purpose

The training loop is responsible for updating adapter weights through gradient descent. PEFT models are fully compatible with standard PyTorch training and the HuggingFace Trainer API.

---

## Using HuggingFace Trainer

### API Signature

```python
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=peft_model,
    args=TrainingArguments(...),
    train_dataset=dataset,
    data_collator=collator,
)
trainer.train()
```

### Recommended Training Arguments

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,  # or bf16=True
    optim="adamw_torch",
)
```

---

## Custom Training Loop

For more control, use a standard PyTorch training loop:

```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Only adapter parameters have gradients
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-4,
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=len(dataloader) * epochs,
)

model.train()
for epoch in range(epochs):
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

---

## Key Considerations

### Only Adapter Weights Are Updated

PEFT automatically freezes base model weights:

```python
# Verify trainable parameters
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

### Memory-Efficient Training

| Technique | Purpose | How to Enable |
|-----------|---------|---------------|
| Gradient checkpointing | Reduce activation memory | `model.gradient_checkpointing_enable()` |
| Mixed precision | Faster, less memory | `fp16=True` in TrainingArguments |
| Gradient accumulation | Larger effective batch | `gradient_accumulation_steps=N` |

### Recommended Hyperparameters for LoRA

| Hyperparameter | Typical Range | Notes |
|----------------|---------------|-------|
| Learning rate | 1e-4 to 3e-4 | Higher than full fine-tuning |
| Batch size | 4-32 | Depends on GPU memory |
| Epochs | 1-5 | LoRA converges quickly |
| Weight decay | 0.0 to 0.1 | Light regularization |

---

## Integration with PEFT

### After Training, Save Adapters

```python
# Training complete
trainer.train()

# Save only adapter weights
model.save_pretrained("./trained-adapter")
```

---

## External Documentation

- **Official Docs**: [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_get_peft_model]] | Creates the trainable model |
| [[huggingface_peft_save_pretrained]] | Saves trained adapters |

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:Wrapper Doc]]
