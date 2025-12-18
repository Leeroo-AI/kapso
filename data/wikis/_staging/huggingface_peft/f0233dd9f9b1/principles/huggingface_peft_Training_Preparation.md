{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|PyTorch Training|https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html]]
* [[source::Doc|Transformers Training|https://huggingface.co/docs/transformers/training]]
|-
! Domains
| [[domain::Training]], [[domain::Model_State]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for preparing a PEFT model for training by setting the correct mode and enabling gradient computation.

=== Description ===

Training Preparation ensures the model is in the correct state before training begins. This includes:
1. Setting training mode (enables dropout, batch norm in train mode)
2. Verifying only adapter parameters have gradients enabled
3. Optionally enabling gradient checkpointing for memory efficiency
4. Setting up optimizer with appropriate learning rates

In PEFT, this step confirms that base model weights are frozen and only adapter weights will be updated.

=== Usage ===

Apply this principle before the training loop:
* **Standard training:** Call `model.train()` to enable training mode
* **Memory optimization:** Enable gradient checkpointing for large models
* **Custom training:** Set up optimizer to target only trainable parameters
* **Verification:** Use `model.print_trainable_parameters()` to confirm setup

== Theoretical Basis ==

'''Training Mode Effects:'''

<syntaxhighlight lang="python">
# Pseudo-code for training mode behavior
def train_mode(model):
    for module in model.modules():
        if isinstance(module, Dropout):
            module.active = True  # Enable dropout
        if isinstance(module, BatchNorm):
            module.track_running_stats = True
</syntaxhighlight>

'''Gradient Configuration in PEFT:'''

After `get_peft_model()`, the gradient state is:
<syntaxhighlight lang="python">
# Pseudo-code for PEFT gradient setup
for name, param in model.named_parameters():
    if "lora_" in name or "modules_to_save" in name:
        param.requires_grad = True   # Adapter weights: trainable
    else:
        param.requires_grad = False  # Base weights: frozen
</syntaxhighlight>

'''Memory-Efficient Training:'''

Gradient checkpointing trades compute for memory:
<syntaxhighlight lang="python">
# With gradient checkpointing
# Forward: activations discarded
# Backward: activations recomputed on-the-fly

model.gradient_checkpointing_enable()
# Now memory = O(sqrt(layers)) instead of O(layers)
</syntaxhighlight>

'''Optimizer Setup:'''

For PEFT models, the optimizer should only receive trainable parameters:
<syntaxhighlight lang="python">
# Filter trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=2e-4)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_model_train_mode]]
