{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|PyTorch|https://pytorch.org/docs/stable/generated/torch.nn.Module.html]]
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::Inference]], [[domain::Model_Serving]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for setting a PEFT model to evaluation mode for inference.

=== Description ===

`model.eval()` is a PyTorch method that sets the model to evaluation mode. For PEFT models, this propagates through both the base model and adapter layers, disabling dropout and setting BatchNorm to use running statistics.

=== Usage ===

Call before running inference on a PEFT model to ensure deterministic outputs and optimal performance.

== Code Reference ==

=== Source Location ===
* '''Library:''' `torch.nn.Module` (PyTorch base)
* '''Method:''' `eval()`

=== Signature ===
<syntaxhighlight lang="python">
def eval(self) -> T:
    """
    Set the module to evaluation mode.

    This is equivalent to self.train(False).

    Returns:
        self: The module in evaluation mode
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# No import needed - method of all nn.Module subclasses
# PeftModel inherits from nn.Module
</syntaxhighlight>

== Usage Examples ==

=== Basic Inference Setup ===
<syntaxhighlight lang="python">
from peft import PeftModel
import torch

# Load model with adapter
model = PeftModel.from_pretrained(base_model, "path/to/adapter")

# Configure for inference
model.eval()

# Run inference without gradient computation
with torch.no_grad():
    outputs = model.generate(input_ids, max_new_tokens=100)
</syntaxhighlight>

=== With Inference Optimizations ===
<syntaxhighlight lang="python">
# Set eval mode and compile for repeated inference
model.eval()
compiled_model = torch.compile(model)

with torch.no_grad():
    # First call triggers compilation
    outputs = compiled_model.generate(input_ids)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Inference_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
