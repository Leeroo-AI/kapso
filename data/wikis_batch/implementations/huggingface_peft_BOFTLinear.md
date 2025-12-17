# Implementation: huggingface_peft_BOFTLinear

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization|https://huggingface.co/papers/2311.06243]]
|-
! Domains
| [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Deep Learning]], [[domain::Linear Layers]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
BOFT implementation for dense linear layers with butterfly factorization-based orthogonal transformations.

=== Description ===
The Linear class implements BOFT (Butterfly Orthogonal Fine-Tuning) specifically for nn.Linear layers. It inherits from both nn.Module and BOFTLayer, providing a complete implementation that combines butterfly factorization with orthogonal transformations for parameter-efficient fine-tuning. During forward passes, it applies rotation matrices computed via Cayley parametrization and butterfly structure to input features before passing them to the base linear layer.

The implementation supports:
* Dynamic computation of butterfly orthogonal matrices during forward pass
* Safe merging of adapter weights into base layer
* Multiplicative dropout for regularization
* Both CUDA-accelerated and CPU fallback modes
* Preservation of dtype throughout computation

=== Usage ===
Use BOFTLinear when fine-tuning linear layers in transformer models, MLPs, or any architecture with nn.Linear layers. It is particularly effective for language models and vision transformers where maintaining orthogonality helps preserve pretrained representations while adapting to new tasks.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/boft/layer.py src/peft/tuners/boft/layer.py]
* '''Lines:''' 466-666

=== Signature ===
<syntaxhighlight lang="python">
class Linear(nn.Module, BOFTLayer):
    """BOFT implemented in a dense layer."""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        boft_block_size: int = 8,
        boft_block_num: int = 0,
        boft_n_butterfly_factor: int = 0,
        boft_dropout: float = 0.1,
        fan_in_fan_out: bool = False,
        init_weights: Union[bool, str] = True,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        """Initialize BOFT Linear layer"""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.boft.layer import Linear
# Or for direct usage:
from peft import get_peft_model, BOFTConfig
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
| boft_block_size || int || No || Block size for BOFT decomposition (default: 8)
|-
| boft_block_num || int || No || Number of blocks (default: 0, auto-calculated)
|-
| boft_n_butterfly_factor || int || No || Butterfly structure depth (default: 0)
|-
| boft_dropout || float || No || Dropout probability for blocks (default: 0.1)
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
| get_delta_weight() || tuple[Tensor, Tensor] || Butterfly rotation matrix and scale parameters
|}

== Core Methods ==

=== forward ===
<syntaxhighlight lang="python">
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """
    Forward pass applying BOFT transformation.

    Process:
    1. Check if adapters are disabled or merged
    2. For each active adapter:
       - Apply Cayley parametrization to get rotation matrices
       - Apply multiplicative dropout
       - Build butterfly block diagonal structure
       - Multiply with permutation matrices
    3. Apply rotation to base layer weights
    4. Apply scaling factors
    5. Compute linear transformation

    Returns output in original dtype.
    """
</syntaxhighlight>

=== merge ===
<syntaxhighlight lang="python">
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    """
    Merge active adapter weights into base layer weights.

    Args:
        safe_merge: If True, check for NaNs before merging
        adapter_names: Specific adapters to merge (None = all active)

    Modifies base_layer.weight in-place.
    """
</syntaxhighlight>

=== unmerge ===
<syntaxhighlight lang="python">
def unmerge(self) -> None:
    """
    Unmerge all merged adapters by applying inverse transformation.

    Uses transpose of rotation matrix and inverse of scaling factor.
    """
</syntaxhighlight>

=== get_delta_weight ===
<syntaxhighlight lang="python">
def get_delta_weight(self, adapter) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the delta weight for the given adapter.

    Returns:
        butterfly_oft_mat: Orthogonal rotation matrix
        boft_s: Scaling parameters
    """
</syntaxhighlight>

== Usage Examples ==

=== Basic Linear Layer Adaptation ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.boft.layer import Linear as BOFTLinear

# Create pretrained linear layer
base_linear = nn.Linear(768, 3072)
# ... load pretrained weights ...

# Add BOFT adapter
boft_linear = BOFTLinear(
    base_layer=base_linear,
    adapter_name="task_adapter",
    boft_block_size=96,  # 768 / 96 = 8 blocks
    boft_block_num=0,
    boft_n_butterfly_factor=1,
    boft_dropout=0.1,
    init_weights=True
)

# Training mode
boft_linear.train()
x = torch.randn(32, 128, 768)  # [batch, seq_len, hidden]
output = boft_linear(x)
print(f"Output shape: {output.shape}")  # [32, 128, 3072]
</syntaxhighlight>

=== Multiple Adapters ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft import get_peft_model, BOFTConfig
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure BOFT
config = BOFTConfig(
    boft_block_size=8,
    boft_n_butterfly_factor=1,
    boft_dropout=0.1,
    target_modules=["c_attn", "c_proj"],
    task_type="CAUSAL_LM"
)

# Create PEFT model with BOFT adapters
peft_model = get_peft_model(model, config)

# Check trainable parameters
peft_model.print_trainable_parameters()
# Output: trainable params: X || all params: Y || trainable%: Z
</syntaxhighlight>

=== Inference with Merged Weights ===
<syntaxhighlight lang="python">
import torch
from peft import PeftModel, BOFTConfig
from transformers import AutoModelForSequenceClassification

# Load model with trained adapter
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model = PeftModel.from_pretrained(base_model, "path/to/boft/adapter")

# Merge for faster inference
model.merge_and_unload()

# Now inference uses merged weights (no adapter overhead)
inputs = torch.randint(0, 30000, (1, 128))
outputs = model(inputs)
print(outputs.logits.shape)
</syntaxhighlight>

=== Custom Block Configuration ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.boft.layer import Linear as BOFTLinear

# Large linear layer
base_layer = nn.Linear(4096, 4096)

# Configure with specific block parameters
boft_linear = BOFTLinear(
    base_layer=base_layer,
    adapter_name="large_model",
    boft_block_size=128,  # 4096 / 128 = 32 blocks
    boft_block_num=0,
    boft_n_butterfly_factor=2,  # More butterfly structure
    boft_dropout=0.05,  # Lower dropout
    init_weights=True
)

# Verify configuration
adapter_name = "large_model"
print(f"Block size: {boft_linear.boft_block_size[adapter_name]}")
print(f"Block num: {boft_linear.boft_block_num[adapter_name]}")
print(f"R shape: {boft_linear.boft_R[adapter_name].shape}")
print(f"s shape: {boft_linear.boft_s[adapter_name].shape}")
</syntaxhighlight>

=== Gradient Checkpointing ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from peft.tuners.boft.layer import Linear as BOFTLinear

base_layer = nn.Linear(2048, 2048)
boft_linear = BOFTLinear(
    base_layer=base_layer,
    adapter_name="checkpoint_demo",
    boft_block_size=128,
    boft_block_num=0,
    boft_n_butterfly_factor=1,
    boft_dropout=0.1,
    init_weights=True
)

# Use gradient checkpointing to save memory during training
x = torch.randn(8, 512, 2048, requires_grad=True)

# Standard forward
# output = boft_linear(x)

# With gradient checkpointing
output = checkpoint(boft_linear, x, use_reentrant=False)

loss = output.sum()
loss.backward()
print(f"Gradient computed with reduced memory usage")
</syntaxhighlight>

== Implementation Details ==

=== Forward Pass Algorithm ===
1. Initialize identity rotation and ones scaling
2. For each active adapter:
   - Reshape boft_R to (N*D, H, H)
   - Apply Cayley transform to get orthogonal blocks
   - Reshape to (N, D, H, H)
   - Apply multiplicative dropout
   - Convert to block diagonal (via CUDA or torch.block_diag)
   - Apply butterfly permutations: P^T * BlockDiag * P
   - Chain multiple butterfly matrices
   - Accumulate rotation and scaling
3. Apply rotation to weight matrix
4. Apply scaling
5. Execute linear transformation

=== Weight Merging Strategy ===
When merging, the transformation is:
* W_merged = (R * W^T)^T * s
* Where R is the butterfly rotation matrix
* And s is the scaling factor

For unmerging:
* W_original = (R^T * W_merged^T)^T / s

=== Memory Efficiency ===
BOFT reduces parameters compared to full fine-tuning:
* Full: O(d_in * d_out)
* BOFT: O(d * log(d) * num_blocks)
* For d=768, block_size=96: ~8 blocks, minimal parameters

=== CUDA Optimization ===
When CUDA extension is available:
* FastBlockDiag provides optimized block diagonal construction
* Significantly faster than torch.block_diag for large tensors
* Falls back gracefully if unavailable

== Related Pages ==
* [[implements::Implementation:huggingface_peft_BOFTLayer]]
* [[uses::Implementation:huggingface_peft_FastBlockDiag]]
* [[uses::Implementation:huggingface_peft_MultiplicativeDropoutLayer]]
* [[related_to::Implementation:huggingface_peft_OFTLinear]]
