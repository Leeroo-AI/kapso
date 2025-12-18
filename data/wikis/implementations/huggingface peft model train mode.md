{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PyTorch Docs|https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train]]
|-
! Domains
| [[domain::Training]], [[domain::Model_State]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for setting the PEFT model to training mode, enabling dropout and gradient computation for adapter training.

=== Description ===

`model.train()` is a PyTorch method that sets the model to training mode. For PEFT models, this enables dropout in LoRA layers and ensures gradients are tracked for the trainable adapter parameters. This is a standard PyTorch API inherited by PeftModel.

=== Usage ===

Call this before starting the training loop. In practice, HuggingFace's `Trainer` handles this automatically, but for custom training loops, explicitly calling `model.train()` is required. Pair with `model.eval()` for inference.

== Code Reference ==

=== Source Location ===
* '''Library:''' [https://pytorch.org/ PyTorch]
* '''Class:''' `torch.nn.Module`
* '''Method:''' `train(mode: bool = True)`

=== Signature ===
<syntaxhighlight lang="python">
def train(self, mode: bool = True) -> T:
    """
    Sets the module in training mode.

    This has effect only on certain modules. E.g., Dropout, BatchNorm, etc.
    are affected by this, and are typically different in training and
    evaluation modes.

    Args:
        mode: Whether to set training mode (True) or evaluation mode (False)

    Returns:
        self: The module itself
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# No explicit import needed - method is on the model
# model.train() is called directly
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| mode || bool || No || Training mode flag. True for training, False for eval. Default: True
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| self || PeftModel || The model in training mode (returns self for chaining)
|}

== Usage Examples ==

=== Basic Training Mode Setup ===
<syntaxhighlight lang="python">
from peft import get_peft_model, LoraConfig

# After creating PEFT model
model = get_peft_model(base_model, lora_config)

# Set to training mode
model.train()

# Now dropout is enabled and gradients tracked
# Ready for training loop
</syntaxhighlight>

=== Custom Training Loop ===
<syntaxhighlight lang="python">
import torch
from torch.optim import AdamW

# Setup
model.train()
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()

    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()
    optimizer.step()

# Switch to eval for validation
model.eval()
with torch.no_grad():
    # ... validation code
</syntaxhighlight>

=== With Gradient Checkpointing ===
<syntaxhighlight lang="python">
# Enable gradient checkpointing before training
model.gradient_checkpointing_enable()

# Set training mode
model.train()

# Now training uses gradient checkpointing to save memory
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Training_Preparation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
