= VBLoRALayer =

== Knowledge Sources ==
* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* [https://huggingface.co/papers/2405.15179 VB-LoRA Paper]
* Source: src/peft/tuners/vblora/layer.py

== Domains ==
* [[Natural Language Processing (NLP)]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Low-Rank Adaptation]]
* [[Neural Network Layers]]
* [[Vector Quantization]]

== Overview ==

=== Description ===
VBLoRALayer is the layer implementation for Vector Bank Low-Rank Adaptation (VB-LoRA). It constructs low-rank adaptation matrices by selecting and combining vectors from a shared vector bank using learnable logits. The layer maintains logits for both A and B matrices (vblora_logits_A and vblora_logits_B) that determine which vectors to select via top-K selection, and the selected vectors are combined to form the low-rank adaptation.

This approach dramatically reduces parameter storage requirements while maintaining adaptation quality, as multiple layers share the same vector bank. The implementation supports both nn.Linear and Conv1D base layers.

=== Usage ===
VBLoRALayer is automatically instantiated when applying VBLoRAConfig to a model. It wraps base layers and adds VB-LoRA adaptation through vector selection and combination. The layer supports merging/unmerging with base weights and handles both training and inference modes, including a special save_only_topk_weights mode for minimal storage.

== Code Reference ==

=== Source Location ===
File: src/peft/tuners/vblora/layer.py
Lines: 27-252

=== Class Signatures ===
<syntaxhighlight lang="python">
class VBLoRALayer(BaseTunerLayer):
    """Base class for VB-LoRA layers"""
    adapter_layer_names = ("vblora_logits_A", "vblora_logits_B", "vblora_vector_bank")

class Linear(nn.Linear, VBLoRALayer):
    """VB-LoRA implementation for dense linear layers"""
    def __init__(
        self,
        base_layer,
        vblora_vector_bank,
        adapter_name: str,
        r: int,
        num_vectors: int,
        vector_length: int,
        topk: int = 2,
        vblora_dropout: float = 0.0,
        init_logits_std: float = 0.01,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs
    ) -> None
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.vblora.layer import VBLoRALayer, Linear
</syntaxhighlight>

== I/O Contract ==

=== VBLoRALayer Attributes ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| vblora_logits_A || nn.ParameterDict || Logits for selecting vectors for A matrix
|-
| vblora_logits_B || nn.ParameterDict || Logits for selecting vectors for B matrix
|-
| vblora_vector_bank || nn.ParameterDict || Shared vector bank across adapters
|-
| r || dict || Dictionary of rank values per adapter
|-
| topk || dict || Dictionary of top-K values per adapter
|-
| vblora_dropout || nn.ModuleDict || Dropout modules per adapter
|-
| merged_adapters || list || List of currently merged adapters
|}

=== Linear Layer Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| base_layer || nn.Module || (required) || Base linear or Conv1D layer
|-
| vblora_vector_bank || nn.ParameterDict || (required) || Shared vector bank
|-
| adapter_name || str || (required) || Name of the adapter
|-
| r || int || (required) || Rank of the adaptation
|-
| num_vectors || int || (required) || Number of vectors in the bank
|-
| vector_length || int || (required) || Length of each vector
|-
| topk || int || 2 || Number of vectors to select (K)
|-
| vblora_dropout || float || 0.0 || Dropout probability
|-
| init_logits_std || float || 0.01 || Std for logits initialization
|-
| fan_in_fan_out || bool || False || Weight storage format
|-
| is_target_conv_1d_layer || bool || False || True if base is Conv1D
|}

=== Key Methods ===
{| class="wikitable"
! Method !! Parameters !! Returns !! Description
|-
| update_layer || adapter_name, vblora_vector_bank, r, topk, num_vectors, vector_length, vblora_dropout, init_logits_std, inference_mode || None || Update layer with new adapter
|-
| merge || safe_merge, adapter_names || None || Merge adapter weights into base
|-
| unmerge || None || None || Unmerge adapter weights
|-
| get_delta_weight || adapter || torch.Tensor || Compute delta weight for adapter
|-
| _get_lora_matrices || adapter, cast_to_fp32 || tuple[Tensor, Tensor] || Get A and B matrices from vector bank
|-
| _get_low_rank_matrix || logits, vblora_vector_bank, topk || torch.Tensor || Select and combine vectors using top-K
|-
| forward || x, *args, **kwargs || torch.Tensor || Forward pass with VB-LoRA
|}

== Usage Examples ==

=== Basic VB-LoRA Model Usage ===
<syntaxhighlight lang="python">
from peft import VBLoRAConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch

# Create model with VB-LoRA
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    topk=2,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(base_model, config)

# Forward pass
input_ids = torch.randint(0, 1000, (2, 10))
outputs = model(input_ids)
</syntaxhighlight>

=== Accessing VB-LoRA Layers ===
<syntaxhighlight lang="python">
# Access a VB-LoRA layer
vblora_layer = model.base_model.model.decoder.layers[0].self_attn.q_proj

# Inspect logits
print(f"Logits A shape: {vblora_layer.vblora_logits_A['default'].shape}")
print(f"Logits B shape: {vblora_layer.vblora_logits_B['default'].shape}")

# Check vector bank
print(f"Vector bank shape: {vblora_layer.vblora_vector_bank['default'].shape}")
</syntaxhighlight>

=== Understanding Vector Selection ===
<syntaxhighlight lang="python">
# Get the low-rank matrices
adapter_name = "default"
A, B = vblora_layer._get_lora_matrices(adapter_name)

print(f"A matrix shape: {A.shape}")  # (rank, in_features)
print(f"B matrix shape: {B.shape}")  # (out_features, rank)

# Compute delta weight
delta = vblora_layer.get_delta_weight(adapter_name)
print(f"Delta weight shape: {delta.shape}")
</syntaxhighlight>

=== Inspecting Top-K Selection ===
<syntaxhighlight lang="python">
import torch.nn.functional as F

# Get logits for A matrix
logits_A = vblora_layer.vblora_logits_A["default"]

# See which vectors are selected (top-k)
topk = vblora_layer.topk["default"]
top_k_logits, indices = logits_A[0, 0].topk(topk)
print(f"Selected vector indices: {indices}")
print(f"Selection weights: {F.softmax(top_k_logits, dim=-1)}")
</syntaxhighlight>

=== Merging and Unmerging ===
<syntaxhighlight lang="python">
# Merge VB-LoRA into base weights
vblora_layer.merge()

print(f"Is merged: {vblora_layer.merged}")

# Unmerge
vblora_layer.unmerge()
</syntaxhighlight>

=== Safe Merging with NaN Check ===
<syntaxhighlight lang="python">
# Merge with safety check
try:
    vblora_layer.merge(safe_merge=True)
    print("Merge successful, no NaNs detected")
except ValueError as e:
    print(f"Merge failed: {e}")
</syntaxhighlight>

=== Training with Dropout ===
<syntaxhighlight lang="python">
# Create model with dropout
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    vblora_dropout=0.1,  # Add dropout
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(base_model, config)

# Dropout is active in training mode
model.train()
output_train = model(input_ids)

# Dropout is disabled in eval mode
model.eval()
output_eval = model(input_ids)
</syntaxhighlight>

=== Inspecting Parameter Efficiency ===
<syntaxhighlight lang="python">
# Check parameter counts
vblora_layer = model.base_model.model.decoder.layers[0].self_attn.q_proj

logits_A_params = vblora_layer.vblora_logits_A["default"].numel()
logits_B_params = vblora_layer.vblora_logits_B["default"].numel()
vector_bank_params = vblora_layer.vblora_vector_bank["default"].numel()

total_vblora_params = logits_A_params + logits_B_params
print(f"Logits A params: {logits_A_params:,}")
print(f"Logits B params: {logits_B_params:,}")
print(f"Total logits: {total_vblora_params:,}")
print(f"Vector bank params: {vector_bank_params:,}")
print(f"Shared across all layers!")
</syntaxhighlight>

=== Handling save_only_topk_weights Mode ===
<syntaxhighlight lang="python">
# After training with save_only_topk_weights=False
model.save_pretrained("./full_vblora")

# For inference-only deployment
config_inference = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    save_only_topk_weights=True  # Minimal storage
)

# Note: Cannot resume training with this mode
# The layer checks for infinity values during training
</syntaxhighlight>

=== Custom Forward Pass ===
<syntaxhighlight lang="python">
# Direct layer usage
x = torch.randn(2, 10, 768)  # (batch, seq, hidden)

# With adapters
vblora_layer.train()
output = vblora_layer(x)

# Disable adapters
vblora_layer.disable_adapters = True
output_base = vblora_layer(x)
</syntaxhighlight>

== Related Pages ==
* [[huggingface_peft_VBLoRAConfig|VBLoRAConfig]] - Configuration class for VB-LoRA
* [[huggingface_peft_VBLoRAModel|VBLoRAModel]] - Model class for VB-LoRA
* [[huggingface_peft_LoraLayer|LoraLayer]] - Standard LoRA layer implementation
* [[Low-Rank Adaptation]]
* [[Vector Quantization]]
* [[Neural Network Layers]]

[[Category:Machine Learning]]
[[Category:PEFT]]
[[Category:Neural Network Layers]]
[[Category:HuggingFace]]
