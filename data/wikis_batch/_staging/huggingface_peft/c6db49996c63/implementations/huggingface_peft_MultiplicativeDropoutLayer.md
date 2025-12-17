# Implementation: huggingface_peft_MultiplicativeDropoutLayer

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization|https://huggingface.co/papers/2311.06243]]
|-
! Domains
| [[domain::Regularization]], [[domain::Deep Learning]], [[domain::Dropout Techniques]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
Multiplicative dropout layer that randomly replaces block matrices with identity matrices for regularization in OFT and BOFT.

=== Description ===
MultiplicativeDropoutLayer implements a specialized dropout technique for block-based orthogonal transformations. Unlike standard dropout that zeros out activations, this layer randomly replaces entire rotation blocks with identity matrices during training. This provides regularization by forcing the model to be robust to missing rotations. The implementation selects a random sample from the batch and randomly drops blocks with probability p.

Key characteristics:
* Drops entire blocks rather than individual elements
* Replaces dropped blocks with identity matrices (not zeros)
* Only active during training mode
* Specialized for block-structured matrices (BOFT/OFT)
* Preserves orthogonality by using identity replacements

=== Usage ===
MultiplicativeDropoutLayer is used internally by BOFTLayer and OFTLayer when module_dropout > 0. It provides regularization for orthogonal fine-tuning methods by randomly disabling rotation blocks. Use higher dropout rates (0.1-0.3) when training data is limited or when you want stronger regularization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File (BOFT):''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/boft/layer.py src/peft/tuners/boft/layer.py]
* '''Lines (BOFT):''' 141-192
* '''File (OFT):''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/layer.py src/peft/tuners/oft/layer.py]
* '''Lines (OFT):''' 28-70

=== Signature ===
<syntaxhighlight lang="python">
class MultiplicativeDropoutLayer(nn.Module):
    """Implements multiplicative dropout for OFT/BOFT."""

    def __init__(self, p=0.0):
        """
        Args:
            p (float): Probability of dropping out a block. Defaults to 0.0.
        """

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor
                - BOFT: shape (N, D, H, H) - batch of block matrices
                - OFT: shape (D, H, H) - single set of blocks
        Returns:
            Tensor: Same shape as input with some blocks replaced by identity
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# BOFT version
from peft.tuners.boft.layer import MultiplicativeDropoutLayer

# OFT version
from peft.tuners.oft.layer import MultiplicativeDropoutLayer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Name !! Type !! Required !! Description
|-
| p || float || No || Dropout probability (default: 0.0)
|-
| x (forward) || torch.Tensor || Yes || Block matrices to apply dropout to
|}

=== Outputs ===
{| class="wikitable"
! Name !! Type !! Description
|-
| forward() output || torch.Tensor || Input with randomly dropped blocks replaced by identity
|}

== Core Methods ==

=== __init__ ===
<syntaxhighlight lang="python">
def __init__(self, p=0.0):
    """
    Initialize dropout layer.

    Args:
        p: Probability of dropping each block [0.0, 1.0]
    """
    super().__init__()
    self.p = p
</syntaxhighlight>

=== forward (BOFT version) ===
<syntaxhighlight lang="python">
def forward(self, x):
    """
    Apply multiplicative dropout to BOFT blocks.

    Process (training mode only):
    1. Select random sample n from batch
    2. Create dropout mask for D blocks
    3. Apply mask only to sample n
    4. Replace masked blocks with identity matrices

    Args:
        x: Tensor of shape (N, D, H, H)
            N = batch size
            D = number of blocks
            H = block size

    Returns:
        Tensor of same shape with some blocks as identity
    """
</syntaxhighlight>

=== forward (OFT version) ===
<syntaxhighlight lang="python">
def forward(self, x):
    """
    Apply multiplicative dropout to OFT blocks.

    Process (training mode and p > 0):
    1. Check if block share (D=1), skip if so
    2. Create dropout mask for D blocks
    3. Replace masked blocks with identity matrices

    Args:
        x: Tensor of shape (D, H, H)
            D = number of blocks
            H = block size

    Returns:
        Tensor of same shape with some blocks as identity
    """
</syntaxhighlight>

== Usage Examples ==

=== Basic Usage (BOFT) ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.boft.layer import MultiplicativeDropoutLayer

# Create dropout layer with 20% drop rate
dropout = MultiplicativeDropoutLayer(p=0.2)

# Create batch of block matrices
N, D, H = 4, 8, 16  # 4 samples, 8 blocks, 16x16 each
blocks = torch.randn(N, D, H, H)

# Training mode - applies dropout
dropout.train()
output_train = dropout(blocks)

# Check which blocks were dropped (should be close to identity)
for i in range(N):
    for j in range(D):
        if torch.allclose(output_train[i, j], torch.eye(H), atol=1e-6):
            print(f"Block [{i}, {j}] was dropped")

# Eval mode - no dropout
dropout.eval()
output_eval = dropout(blocks)
assert torch.equal(output_eval, blocks), "Should be unchanged in eval mode"
</syntaxhighlight>

=== Usage in BOFT Forward Pass ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.boft.layer import Linear as BOFTLinear

# Create BOFT layer with dropout
base_layer = nn.Linear(768, 768)
boft_layer = BOFTLinear(
    base_layer=base_layer,
    adapter_name="with_dropout",
    boft_block_size=96,
    boft_block_num=0,
    boft_n_butterfly_factor=1,
    boft_dropout=0.15,  # 15% dropout
    init_weights=True
)

# Dropout is applied automatically during forward pass
boft_layer.train()
x = torch.randn(32, 768)
output = boft_layer(x)

# Check dropout layer
dropout_layer = boft_layer.boft_dropout["with_dropout"]
print(f"Dropout probability: {dropout_layer.p}")  # 0.15
print(f"Is training: {dropout_layer.training}")  # True
</syntaxhighlight>

=== Usage in OFT Forward Pass ===
<syntaxhighlight lang="python">
import torch
import torch.nn as nn
from peft.tuners.oft.layer import Linear as OFTLinear

# Create OFT layer with dropout
base_layer = nn.Linear(512, 512)
oft_layer = OFTLinear(
    base_layer=base_layer,
    adapter_name="with_dropout",
    r=8,
    oft_block_size=0,
    module_dropout=0.1,  # 10% dropout
    init_weights=True
)

# Training with dropout
oft_layer.train()
x = torch.randn(16, 512)
output_train = oft_layer(x)

# Inference without dropout
oft_layer.eval()
output_eval = oft_layer(x)

# Outputs will differ due to dropout
print(f"Train output mean: {output_train.mean():.6f}")
print(f"Eval output mean: {output_eval.mean():.6f}")
</syntaxhighlight>

=== Comparing Dropout Rates ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.boft.layer import MultiplicativeDropoutLayer

# Different dropout rates
dropouts = {
    "none": MultiplicativeDropoutLayer(p=0.0),
    "low": MultiplicativeDropoutLayer(p=0.1),
    "medium": MultiplicativeDropoutLayer(p=0.2),
    "high": MultiplicativeDropoutLayer(p=0.3),
}

# Set all to training mode
for d in dropouts.values():
    d.train()

# Test on same input
N, D, H = 1, 100, 8
blocks = torch.randn(N, D, H, H)

results = {}
for name, dropout in dropouts.items():
    output = dropout(blocks)

    # Count identity blocks
    identity = torch.eye(H)
    num_dropped = 0
    for j in range(D):
        if torch.allclose(output[0, j], identity, atol=1e-6):
            num_dropped += 1

    results[name] = num_dropped / D
    print(f"{name}: {num_dropped}/{D} blocks dropped ({results[name]:.1%})")

# Results should approximately match configured probabilities
# none: ~0%, low: ~10%, medium: ~20%, high: ~30%
</syntaxhighlight>

=== Identity Verification ===
<function_calls>
import torch
from peft.tuners.oft.layer import MultiplicativeDropoutLayer

# Create dropout layer
dropout = MultiplicativeDropoutLayer(p=0.5)
dropout.train()

# Create blocks
D, H = 10, 16
blocks = torch.randn(D, H, H)

# Apply dropout multiple times
for trial in range(5):
    output = dropout(blocks)

    # Verify dropped blocks are identity
    identity = torch.eye(H)
    for j in range(D):
        block = output[j]
        is_identity = torch.allclose(block, identity, atol=1e-6)
        is_original = torch.allclose(block, blocks[j], atol=1e-6)

        assert is_identity or is_original, f"Block {j} is neither identity nor original!"

        if is_identity:
            # Verify it's exactly identity (orthogonal, determinant = 1)
            assert torch.allclose(block @ block.T, identity, atol=1e-5)
            det = torch.linalg.det(block)
            assert abs(det - 1.0) < 1e-5, f"Identity determinant: {det}"

    print(f"Trial {trial+1}: All dropped blocks verified as identity")
</syntaxhighlight>

=== Block Share Behavior (OFT) ===
<syntaxhighlight lang="python">
import torch
from peft.tuners.oft.layer import MultiplicativeDropoutLayer

dropout = MultiplicativeDropoutLayer(p=0.3)
dropout.train()

# Single block (block_share=True case)
single_block = torch.randn(1, 8, 8)
output_single = dropout(single_block)

# With D=1, dropout is skipped
assert torch.equal(output_single, single_block), "Should skip dropout for block_share"

# Multiple blocks
multi_block = torch.randn(10, 8, 8)
output_multi = dropout(multi_block)

# Dropout should be applied
num_dropped = sum(
    torch.allclose(output_multi[i], torch.eye(8), atol=1e-6)
    for i in range(10)
)
print(f"Single block: unchanged (block_share)")
print(f"Multiple blocks: {num_dropped}/10 dropped")
</syntaxhighlight>

== Implementation Details ==

=== BOFT vs OFT Differences ===
BOFT version:
* Input shape: (N, D, H, H) with batch dimension
* Selects random sample from batch to apply dropout
* Creates full mask for all N samples but only applies to one

OFT version:
* Input shape: (D, H, H) without batch dimension
* Applies dropout to all blocks
* Skips if D=1 (block_share mode)

=== Dropout Mechanism ===
Unlike standard dropout (multiply by 0), multiplicative dropout:
1. Creates binary mask [1, 1, 0, 1, 0, 1, ...]
2. Where mask=1: keep original block
3. Where mask=0: replace with identity matrix

This preserves orthogonality since identity is orthogonal.

=== Probability Calculation ===
For dropout probability p and D blocks:
```python
num_to_replace = int(p * D)
num_zeros = D - num_to_replace
mask = cat([ones(num_to_replace), zeros(num_zeros)])
mask = mask[randperm(D)]  # Shuffle
```

=== Training vs Eval Mode ===
* Training: applies dropout with probability p
* Eval: returns input unchanged (no dropout)

This is standard PyTorch behavior via `self.training` check.

=== Memory Efficiency ===
Identity matrices are created on-the-fly:
```python
eye_matrix = torch.eye(H, device=x.device)
```

Not stored persistently, only during forward pass.

=== Gradient Flow ===
When block is replaced with identity:
* Forward: block → identity
* Backward: gradients flow through identity
* Original block parameters get zero gradient for that sample

This encourages robustness to missing blocks.

=== Random Sample Selection (BOFT) ===
BOFT applies dropout to only one random sample per batch:
```python
n_random = torch.randint(0, N, (1,)).item()
```

This reduces variance compared to dropping in all samples.

== Related Pages ==
* [[used_by::Implementation:huggingface_peft_BOFTLayer]]
* [[used_by::Implementation:huggingface_peft_OFTLayer]]
