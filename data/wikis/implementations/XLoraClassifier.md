= XLoraClassifier =

== Knowledge Sources ==

* '''Repository''': [https://github.com/huggingface/peft HuggingFace PEFT]
* '''Paper''': X-LoRA: Mixture of Low-Rank Adapters
* '''Type''': Classifier Module
* '''Module''': peft.tuners.xlora.classifier

== Domains ==

[[Category:Natural_Language_Processing]]
[[Category:Parameter_Efficient_Fine_Tuning]]
[[Category:Mixture_of_Experts]]
[[Category:Low_Rank_Adaptation]]
[[Category:Neural_Networks]]

== Overview ==

=== Description ===

XLoraClassifier is a neural network classifier that dynamically selects and weights multiple LoRA adapters for the X-LoRA (Mixture of LoRA Experts) technique. It takes hidden states from the base model and predicts scaling values (weights) for each LoRA adapter at each layer and position in the sequence.

The classifier acts as a gating mechanism in a mixture-of-experts setup, where each LoRA adapter is an expert. It can operate in two modes:
* '''Dense mode''': Uses softmax to produce normalized weights across all adapters
* '''Sparse mode''': Selects top-k adapters and optionally applies softmax over them

The classifier supports configurable depth (number of hidden layers), size, dropout, and temperature-scaled softmax for controlling prediction sharpness.

=== Usage ===

XLoraClassifier is used internally by XLoraModel to determine adapter mixing weights dynamically based on input. It performs two forward passes: a "scaling pass" with dummy scalings to get logits, then a "real pass" with predicted scalings.

== Code Reference ==

=== Source Location ===

<code>/tmp/praxium_repo_zyf9ywdz/src/peft/tuners/xlora/classifier.py</code>

=== Signature ===

<syntaxhighlight lang="python">
class XLoraClassifier(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        config: XLoraConfig,
        n_classes: int,
        n_layers: int,
        device: torch.device
    )

    def forward(
        self,
        result,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from peft.tuners.xlora.classifier import XLoraClassifier
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| model || nn.Module || Required || PeftModel containing LoRA adapters
|-
| config || XLoraConfig || Required || Configuration for X-LoRA
|-
| n_classes || int || Required || Number of LoRA adapters (experts)
|-
| n_layers || int || Required || Number of LoRA adapter layers
|-
| device || torch.device || Required || Device to place classifier on
|}

=== Forward Method Parameters ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| result || ModelOutput || Required || Model output containing hidden_states
|-
| input_ids || torch.LongTensor || None || Input token IDs (batch_size, seq_len)
|-
| inputs_embeds || torch.FloatTensor || None || Input embeddings (batch_size, seq_len, hidden_size)
|}

=== Return Values ===

{| class="wikitable"
! Method !! Return Type !! Shape !! Description
|-
| forward || torch.Tensor || (batch_size, seq_len, n_layers, n_classes) || Scaling weights for each adapter
|-
| make_dummy_scalings || torch.Tensor || (batch_size, seq_len, n_layers, n_classes) || Dummy scalings for scaling pass
|}

=== Output Tensor Details ===

The output tensor has shape <code>(batch_size, seq_len, n_layers, n_classes)</code>:
* '''batch_size''': Number of sequences in batch
* '''seq_len''': Sequence length
* '''n_layers''': Number of LoRA adapter layers in model
* '''n_classes''': Number of LoRA adapters (experts)

If <code>enable_softmax=True</code>, values are softmax-normalized across the n_classes dimension, summing to 1.

== Usage Examples ==

=== Basic Classifier Setup ===

<syntaxhighlight lang="python">
import torch
from peft import XLoraConfig
from peft.tuners.xlora.classifier import XLoraClassifier

# Configuration
config = XLoraConfig(
    hidden_size=4096,
    xlora_depth=2,
    xlora_size=2048,
    xlora_dropout_p=0.1,
    softmax_temperature=1.0,
    enable_softmax=True
)

# Create classifier
n_adapters = 4  # Number of LoRA experts
n_layers = 32   # Number of adapter layers
device = torch.device("cuda")

classifier = XLoraClassifier(
    model=peft_model,
    config=config,
    n_classes=n_adapters,
    n_layers=n_layers,
    device=device
)
</syntaxhighlight>

=== Forward Pass Example ===

<syntaxhighlight lang="python">
import torch

# Prepare inputs
input_ids = torch.randint(0, 50000, (2, 128))  # batch=2, seq_len=128

# Get model output with hidden states
result = model(
    input_ids,
    output_hidden_states=True,
    return_dict=True
)

# Get scalings from classifier
scalings = classifier(result, input_ids=input_ids)
# scalings.shape = (2, 128, 32, 4) for batch=2, seq=128, layers=32, adapters=4

print(f"Scalings shape: {scalings.shape}")
print(f"Scalings sum per position: {scalings[0, 0, 0].sum()}")  # Should be ~1.0 with softmax
</syntaxhighlight>

=== Shallow vs Deep Classifier ===

<syntaxhighlight lang="python">
from peft import XLoraConfig
from peft.tuners.xlora.classifier import XLoraClassifier

# Shallow classifier (single layer, fast)
shallow_config = XLoraConfig(
    hidden_size=4096,
    xlora_depth=1,  # Single linear layer
    enable_softmax=True
)

shallow_classifier = XLoraClassifier(
    model=model,
    config=shallow_config,
    n_classes=4,
    n_layers=32,
    device=device
)

# Deep classifier (multiple layers, more capacity)
deep_config = XLoraConfig(
    hidden_size=4096,
    xlora_depth=4,  # Multiple hidden layers
    xlora_size=2048,
    xlora_dropout_p=0.2,
    enable_softmax=True
)

deep_classifier = XLoraClassifier(
    model=model,
    config=deep_config,
    n_classes=4,
    n_layers=32,
    device=device
)
</syntaxhighlight>

=== Temperature-Scaled Softmax ===

<syntaxhighlight lang="python">
from peft import XLoraConfig

# Sharp predictions (low temperature)
sharp_config = XLoraConfig(
    hidden_size=4096,
    xlora_depth=2,
    softmax_temperature=0.5,  # Lower = sharper, more decisive
    enable_softmax=True
)

# Smooth predictions (high temperature)
smooth_config = XLoraConfig(
    hidden_size=4096,
    xlora_depth=2,
    softmax_temperature=2.0,  # Higher = smoother, more uniform
    enable_softmax=True
)

classifier_sharp = XLoraClassifier(model, sharp_config, 4, 32, device)
classifier_smooth = XLoraClassifier(model, smooth_config, 4, 32, device)

# With temperature=0.5, output more concentrated on best adapter
# With temperature=2.0, output more evenly distributed across adapters
</syntaxhighlight>

=== Layerwise vs Shared Scalings ===

<syntaxhighlight lang="python">
from peft import XLoraConfig

# Layerwise scalings (different weights per layer)
layerwise_config = XLoraConfig(
    hidden_size=4096,
    xlora_depth=2,
    layerwise_scalings=True,  # Predict per-layer weights
    enable_softmax=True
)

# Shared scalings (same weights broadcast to all layers)
shared_config = XLoraConfig(
    hidden_size=4096,
    xlora_depth=2,
    layerwise_scalings=False,  # Single set of weights for all layers
    enable_softmax=True
)

# Layerwise has more parameters but allows layer-specific adapter selection
layerwise_classifier = XLoraClassifier(model, layerwise_config, 4, 32, device)

# Shared has fewer parameters and assumes consistent adapter importance
shared_classifier = XLoraClassifier(model, shared_config, 4, 32, device)
</syntaxhighlight>

=== Dummy Scalings for Initial Pass ===

<syntaxhighlight lang="python">
import torch

# Create dummy scalings for the first forward pass
# (needed to get hidden states for actual classification)
input_ids = torch.randint(0, 50000, (2, 64))

dummy_scalings = classifier.make_dummy_scalings(input_ids=input_ids)
# Shape: (2, 64, n_layers, n_classes)
# Filled with scaling_pass_value (typically 0 or 1/n_classes)

print(f"Dummy scalings shape: {dummy_scalings.shape}")
print(f"Dummy value: {dummy_scalings[0, 0, 0, 0].item()}")
</syntaxhighlight>

=== Logging Scalings for Analysis ===

<syntaxhighlight lang="python">
# Enable scalings logging
classifier.scalings_logging = True

# Run multiple forward passes
for batch in dataloader:
    result = model(batch["input_ids"], output_hidden_states=True)
    scalings = classifier(result, input_ids=batch["input_ids"])

# Access logged scalings
all_scalings = classifier.log_scalings
print(f"Logged {len(all_scalings)} scaling tensors")

# Analyze adapter usage
for i, scalings in enumerate(all_scalings):
    mean_weights = scalings.mean(dim=(0, 1, 2))  # Average across batch, seq, layers
    print(f"Batch {i} adapter weights: {mean_weights}")

# Get bucketed scalings by sequence length
bucketed = classifier._get_bucketed_scalings()
for seq_len, (positions, tensors) in bucketed.items():
    print(f"Sequence length {seq_len}: {len(tensors)} samples")
</syntaxhighlight>

=== Override Scaling Pass Value ===

<syntaxhighlight lang="python">
# Set custom scaling pass value
classifier._set_override_scaling_pass_value(0.25)

# Or reset to default (1/n_classes)
classifier._set_override_scaling_pass_value(None)

# This affects dummy_scalings used in the initial pass
dummy = classifier.make_dummy_scalings(input_ids=input_ids)
print(f"Override value: {dummy[0, 0, 0, 0].item()}")
</syntaxhighlight>

== Implementation Details ==

=== Architecture ===

The classifier network structure depends on <code>xlora_depth</code>:

'''Depth = 1''' (Shallow):
<syntaxhighlight lang="python">
Linear(hidden_size -> n_classes * n_layers)  # if layerwise_scalings
Linear(hidden_size -> n_classes)             # if not layerwise_scalings
</syntaxhighlight>

'''Depth > 1''' (Deep):
<syntaxhighlight lang="python">
Linear(hidden_size -> xlora_size)
ReLU()
Dropout(p=xlora_dropout_p)  # if dropout > 0
Linear(xlora_size -> xlora_size)  # repeated (depth-2) times
ReLU()
Dropout(p=xlora_dropout_p)
Linear(xlora_size -> n_classes * n_layers)  # or n_classes
</syntaxhighlight>

=== Temperature Scaling ===

<syntaxhighlight lang="python">
class TemperatureScaledSoftmax(nn.Module):
    def forward(self, logits):
        scaled_logits = logits / self.temperature
        return softmax(scaled_logits)
</syntaxhighlight>

Lower temperature (<1.0) makes distribution sharper, higher temperature (>1.0) makes it smoother.

== Related Pages ==

* [[huggingface_peft_XLoraConfig|XLoraConfig]] - Configuration for X-LoRA
* [[huggingface_peft_XLoraModel|XLoraModel]] - Main X-LoRA model class
* [[huggingface_peft_XLoraLayer|XLoraLayer]] - Layer implementation
* [[huggingface_peft_LoraModel|LoraModel]] - Base LoRA implementation
* [[Mixture_of_Experts|MoE Architectures]]
* [[Parameter_Efficient_Fine_Tuning|PEFT Overview]]
