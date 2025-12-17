# Implementation: huggingface_peft_OFTLinear

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|Controlling Text-to-Image Diffusion by Orthogonal Finetuning|https://arxiv.org/abs/2306.07280]]
|-
! Domains
| [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Deep Learning]], [[domain::Linear Layers]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
OFT implementation for dense linear layers with block-diagonal orthogonal transformations.

=== Description ===
The Linear class implements OFT (Orthogonal Fine-Tuning) specifically for nn.Linear layers. It inherits from both nn.Module and OFTLayer, providing a complete implementation that applies block-diagonal orthogonal transformations to input features. Unlike BOFT which uses butterfly factorization, OFT directly constructs and applies orthogonal blocks through the OFTRotationModule. The forward pass applies the rotation module directly to inputs before passing them to the base linear layer.

Key features:
* Direct application of orthogonal rotations via OFTRotationModule
* Support for Cayley and Cayley-Neumann parametrization
* Optional COFT (constrained OFT) for controlled rotation freedom
* Block sharing for parameter efficiency
* Clean merging/unmerging without butterfly structure overhead

=== Usage ===
Use OFTLinear when fine-tuning linear layers in transformer models, MLPs, or any architecture with nn.Linear layers. Particularly effective when you need simpler orthogonal transformations without the complexity of butterfly factorization. Suitable for diffusion models, vision transformers, and language models where maintaining orthogonality is beneficial.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/layer.py src/peft/tuners/oft/layer.py]
* '''Lines:''' 526-676

=== Signature ===
<syntaxhighlight lang="python">
class Linear(nn.Module, OFTLayer):
    """OFT implemented in Linear layer"""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 8,
        oft_block_size: int = 0,
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        fan_in_fan_out: bool = False,
        init_weights: Union[bool, str] = True,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        """Initialize OFT Linear layer"""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.oft.layer import Linear
# Or for direct usage:
from peft import get_peft_model, OFTConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Linear || Yes || The pretrained linear layer to adapt
|-
| adapter_name || str || Yes || Name identifier for this adapter
|-
| r || int || No || Number of OFT blocks (default: 8)
|-
| oft_block_size || int || No || Block size (default: 0, auto-calculated)
|-
| module_dropout || float || No || Dropout probability for blocks (default: 0.0)
|-
| coft || bool || No || Use constrained OFT (default: False)
|-
| eps || float || No || COFT constraint strength (default: 6e-5)
|-
| block_share || bool || No || Share parameters across blocks (default: False)
|-
| use_cayley_neumann || bool || No || Use approximation (default: False)
|-
| num_cayley_neumann_terms || int || No || Approximation terms (default: 5)
|-
| fan_in_fan_out || bool || No || Weight storage format flag (default: False)
|-
| init_weights || bool/str || No || Initialization strategy (default: True)
|-
| is_target_conv_1d_layer || bool || No || Conv1d compatibility flag (default: False)
|}

=== Outputs ===
{| class="wikitable"
! Name !! Type !! Description
|-
| forward() output || torch.Tensor || Transformed features with same shape as input
|-
| get_delta_weight() || torch.Tensor || Orthogonal rotation matrix
|}

== Core Methods ==

=== forward ===
<syntaxhighlight lang="python">
def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Forward pass applying OFT transformation.

    Process:
    1. Check if adapters are disabled or merged
    2. For each active adapter:
       - Cast input to rotation module dtype
       - Apply OFTRotationModule (includes dropout if configured)
    3. Pass rotated input to base layer
    4. Return result in original dtype

    Returns output with rotation applied to features.
    """
</syntaxhighlight>

=== merge ===
<syntaxhighlight lang="python">
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    """
    Merge active adapter weights into base layer weights.

    Process:
    1. Get orthogonal matrix from OFTRotationModule
    2. Apply rotation to base layer weights: W_new = R @ W^T
    3. Transpose back and update base_layer.weight

    Args:
        safe_merge: If True, check for NaNs before merging
        adapter_names: Specific adapters to merge (None = all active)
    """
</syntaxhighlight>

=== unmerge ===
<syntaxhighlight lang="python">
def unmerge(self) -> None:
    """
    Unmerge all merged adapters by applying inverse transformation.

    Uses matrix inverse: W_original = R^-1 @ W_merged^T
    Computes inverse in float32 for numerical stability.
    """
</syntaxhighlight>

=== get_delta_weight ===
<syntaxhighlight lang="python">
def get_delta_weight(self, adapter_name) -> torch.Tensor:
    """
    Get orthogonal rotation matrix for given adapter.

    Calls oft_R[adapter_name].get_weight() to obtain
    full block-diagonal orthogonal matrix.

    Returns:
        torch.Tensor: Orthogonal matrix [in_features, in_features]
    """
</syntaxhighlight>

== Usage Examples ==

=== Basic OFT Linear Layer ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Linear as OFTLinear

# Create pretrained linear layer
base_linear = nn.Linear(768, 3072)
# ... load pretrained weights ...

# Add OFT adapter
oft_linear = OFTLinear(
    base_layer=base_linear,
    adapter_name="task_adapter",
    r=8,  # 8 blocks, block_size = 768/8 = 96
    oft_block_size=0,
    module_dropout=0.0,
    coft=False,
    block_share=False,
    init_weights=True
)

# Training mode
oft_linear.train()
x = torch.randn(32, 128, 768)  # [batch, seq_len, hidden]
output = oft_linear(x)
print(f"Output shape: {output.shape}")  # [32, 128, 3072]
</syntaxhighlight>

=== Transformer with OFT ===
<syntaxhighlight lang="python">
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, OFTConfig

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure OFT for attention projections
config = OFTConfig(
    r=8,
    module_dropout=0.0,
    coft=False,
    block_share=False,
    target_modules=["c_attn", "c_proj"],  # Query, key, value, and output
    task_type="CAUSAL_LM"
)

# Create PEFT model with OFT adapters
peft_model = get_peft_model(model, config)

# Check trainable parameters
peft_model.print_trainable_parameters()
# trainable params: ~500K || all params: 124M || trainable%: 0.4

# Fine-tune the model
# ... training loop ...
</syntaxhighlight>

=== Using COFT for Controlled Rotation ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Linear as OFTLinear

base_layer = nn.Linear(1024, 1024)

# Standard OFT - no constraints
oft_standard = OFTLinear(
    base_layer=base_layer,
    adapter_name="standard",
    r=16,
    coft=False,
    init_weights=True
)

# COFT - constrained rotation
oft_coft = OFTLinear(
    base_layer=base_layer,
    adapter_name="coft",
    r=16,
    coft=True,  # Enable constraints
    eps=1e-4,  # Control rotation freedom
    init_weights=True
)

# Test rotation matrices
R_standard = oft_standard.get_delta_weight("standard")
R_coft = oft_coft.get_delta_weight("coft")

# COFT matrix should be closer to identity
identity = torch.eye(1024)
dist_standard = torch.norm(R_standard - identity)
dist_coft = torch.norm(R_coft - identity)

print(f"Standard OFT distance from I: {dist_standard:.4f}")
print(f"COFT distance from I: {dist_coft:.4f}")
# COFT should have smaller distance
</syntaxhighlight>

=== Block Sharing for Efficiency ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Linear as OFTLinear

base_layer = nn.Linear(2048, 2048)

# Without block sharing
oft_no_share = OFTLinear(
    base_layer=base_layer,
    adapter_name="no_share",
    r=32,  # 32 different blocks
    oft_block_size=0,
    block_share=False,
    init_weights=True
)

# With block sharing
oft_share = OFTLinear(
    base_layer=base_layer,
    adapter_name="share",
    r=32,  # Same r, but shared block
    oft_block_size=0,
    block_share=True,
    init_weights=True
)

# Parameter counts
no_share_params = oft_no_share.oft_R["no_share"].weight.numel()
share_params = oft_share.oft_R["share"].weight.numel()

print(f"Without sharing: {no_share_params} parameters")
print(f"With sharing: {share_params} parameters")
print(f"Reduction: {no_share_params / share_params:.1f}x")

# Forward pass produces same shape outputs
x = torch.randn(4, 2048)
out_no_share = oft_no_share(x)
out_share = oft_share(x)
print(f"Outputs: {out_no_share.shape}, {out_share.shape}")
</syntaxhighlight>

=== Merging for Production ===
<syntaxhighlight lang="python">
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

# Load model with trained adapter
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model = PeftModel.from_pretrained(base_model, "path/to/oft/adapter")

print("Before merge:")
print(f"Active adapters: {model.active_adapters}")
print(f"Merged: {model.base_model.model.bert.encoder.layer[0].attention.self.query.merged}")

# Merge for faster inference
model.merge_and_unload()

print("\nAfter merge:")
# Now it's just the base model with integrated weights
inputs = torch.randint(0, 30000, (1, 128))
outputs = model(inputs)
print(f"Output shape: {outputs.logits.shape}")
</syntaxhighlight>

=== Comparing Cayley Methods ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
import time
from peft.tuners.oft.layer import Linear as OFTLinear

base_layer = nn.Linear(1024, 1024)

# Exact Cayley (matrix solve)
oft_exact = OFTLinear(
    base_layer=base_layer,
    adapter_name="exact",
    r=16,
    use_cayley_neumann=False,  # Use exact solve
    init_weights=True
)

# Cayley-Neumann approximation
oft_approx = OFTLinear(
    base_layer=base_layer,
    adapter_name="approx",
    r=16,
    use_cayley_neumann=True,  # Use series
    num_cayley_neumann_terms=5,
    init_weights=True
)

# Benchmark
x = torch.randn(64, 1024)

start = time.time()
for _ in range(100):
    _ = oft_exact(x)
exact_time = time.time() - start

start = time.time()
for _ in range(100):
    _ = oft_approx(x)
approx_time = time.time() - start

print(f"Exact Cayley: {exact_time:.3f}s")
print(f"Cayley-Neumann: {approx_time:.3f}s")
print(f"Speedup: {exact_time / approx_time:.2f}x")
</syntaxhighlight>

=== Multi-Adapter Setup ===
<syntaxhighlight lang="python">
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, OFTConfig, PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Create first adapter for task A
config_a = OFTConfig(
    r=8,
    target_modules=["c_attn"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config_a, adapter_name="task_a")

# Add second adapter for task B
config_b = OFTConfig(
    r=16,
    target_modules=["c_attn"],
    task_type="CAUSAL_LM"
)
model.add_adapter("task_b", config_b)

# Switch between adapters
model.set_adapter("task_a")
inputs = torch.tensor([[1, 2, 3]])
output_a = model(inputs)

model.set_adapter("task_b")
output_b = model(inputs)

print(f"Task A logits: {output_a.logits.shape}")
print(f"Task B logits: {output_b.logits.shape}")

# Save both adapters
model.save_pretrained("./oft_multi_adapter")
</syntaxhighlight>

== Implementation Details ==

=== Rotation Application ===
Unlike BOFT, OFT's forward pass is simpler:
1. Cast input to rotation module dtype
2. Call oft_R(x) which handles all rotation logic
3. Pass rotated features to base layer
4. Return in original dtype

No butterfly operations or permutation matrices needed.

=== Merging Strategy ===
Merging applies rotation to weights directly:
```python
oft_mat = self.get_delta_weight(adapter_name)
orig_weights = base_layer.weight.data  # [out_features, in_features]
orig_weights = torch.transpose(orig_weights, 0, 1)
orig_weights = torch.mm(oft_mat, orig_weights)
orig_weights = torch.transpose(orig_weights, 0, 1)
```

Result: W_merged = (R @ W^T)^T = W @ R^T

=== Unmerging with Inverse ===
Unmerging requires matrix inversion:
```python
oft_mat = self.get_delta_weight(adapter_name)
oft_mat = oft_mat.to(torch.float32)  # For numerical stability
orig_weights = torch.mm(
    torch.linalg.inv(oft_mat),
    orig_weights.to(torch.float32)
)
```

Float32 used even if model is FP16 for stable inversion.

=== Parameter Efficiency ===
OFT parameters per adapter:
* Without block sharing: r * n_elements where n_elements = block_size * (block_size - 1) / 2
* With block sharing: 1 * n_elements (32x reduction for r=32)
* Example: For d=768, r=8, block_size=96: 8 * 4560 = 36,480 parameters

Much smaller than LoRA rank * 2 * d or full fine-tuning.

=== Orthogonality Guarantees ===
OFTRotationModule ensures orthogonality through:
1. Skew-symmetric parametrization
2. Cayley transform properties
3. Block diagonal structure preserves orthogonality

Result: R^T @ R = I (up to numerical precision)

=== Module Dropout Integration ===
When module_dropout > 0:
* MultiplicativeDropoutLayer applied in OFTRotationModule.forward()
* Randomly replaces blocks with identity during training
* No dropout during eval mode

=== Conv1d Compatibility ===
The is_target_conv_1d_layer flag enables usage with:
* transformers Conv1D layers
* 1D convolutional architectures
* Handles weight transpose conventions

== Related Pages ==
* [[implements::Implementation:huggingface_peft_OFTLayer]]
* [[uses::Implementation:huggingface_peft_OFTRotationModule]]
* [[uses::Implementation:huggingface_peft_MultiplicativeDropoutLayer]]
* [[alternative_to::Implementation:huggingface_peft_BOFTLinear]]
* [[related_to::Implementation:huggingface_peft_OFTConv2d]]
