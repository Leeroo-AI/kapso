# Implementation: huggingface_peft_Training_Loop

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Trainer|https://huggingface.co/docs/transformers/main_classes/trainer]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Training]], [[domain::Transformers]], [[domain::External]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | Wrapper Doc |
| **Source** | External (transformers library) |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_Training_Loop]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

Training a PEFT model uses standard PyTorch training loops or the HuggingFace `Trainer`. Since PEFT models are standard PyTorch modules, any training infrastructure works. Only adapter parameters (with `requires_grad=True`) are updated during backpropagation.

== Training Options ==

=== Option 1: HuggingFace Trainer ===

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()
```

=== Option 2: Custom PyTorch Loop ===

```python
from torch.optim import AdamW

optimizer = AdamW(peft_model.parameters(), lr=2e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = peft_model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

=== Option 3: Accelerate ===

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(
    peft_model, optimizer, dataloader
)

for batch in dataloader:
    outputs = model(**batch)
    accelerator.backward(outputs.loss)
    optimizer.step()
    optimizer.zero_grad()
```

== Key Considerations ==

| Aspect | Recommendation |
|--------|----------------|
| Learning Rate | 2e-4 to 1e-3 (higher than full fine-tuning) |
| Batch Size | Use gradient accumulation for larger effective batch |
| Optimizer | AdamW with default betas |
| Scheduler | Cosine or linear decay |

== Related Functions ==

* [[huggingface_peft_get_peft_model]] - Create trainable PEFT model
* [[huggingface_peft_save_pretrained]] - Save after training

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Adapter_Training]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]

[[Category:Implementation]]
[[Category:Wrapper_Doc]]
[[Category:Training]]
