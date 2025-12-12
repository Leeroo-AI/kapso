{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Mixed Precision Training|https://arxiv.org/abs/1710.03740]]
* [[source::Doc|NVIDIA Mixed Precision|https://developer.nvidia.com/automatic-mixed-precision]]
* [[source::Doc|PyTorch AMP|https://pytorch.org/docs/stable/amp.html]]
|-
! Domains
| [[domain::Optimization]], [[domain::Training]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Training technique that uses lower precision (FP16/BF16) for most operations while maintaining FP32 for critical computations, reducing memory and increasing speed.

=== Description ===
Mixed Precision Training combines different numerical precisions to optimize training efficiency. Forward and backward passes use FP16 or BF16 (16-bit) for faster computation and reduced memory, while master weights and certain sensitive operations remain in FP32 for numerical stability. Modern GPUs have specialized Tensor Cores that accelerate 16-bit operations, providing 2-3x speedup.

=== Usage ===
Use this principle to accelerate training and reduce memory usage on modern GPUs (Volta and newer). BF16 is preferred for LLM training due to its larger exponent range preventing overflow. Unsloth automatically handles mixed precision through its optimizations.

== Theoretical Basis ==
'''Precision Formats:'''
{| class="wikitable"
! Format !! Bits !! Exponent !! Mantissa !! Range !! Use
|-
|| FP32 || 32 || 8 || 23 || ±3.4e38 || Master weights
|-
|| FP16 || 16 || 5 || 10 || ±65504 || Training (with scaling)
|-
|| BF16 || 16 || 8 || 7 || ±3.4e38 || Training (preferred)
|}

'''Why Mixed Precision Works:'''
1. Neural networks are robust to precision reduction
2. Gradients don't need full precision for direction
3. Master weights maintain accuracy for accumulation

'''Training Pipeline:'''
<syntaxhighlight lang="python">
# Mixed precision training with gradient scaling
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # For FP16 (not needed for BF16)

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in FP16/BF16
    with autocast(dtype=torch.float16):  # or bfloat16
        outputs = model(batch)
        loss = criterion(outputs)
    
    # Backward with scaling (prevents underflow in FP16)
    scaler.scale(loss).backward()
    
    # Unscale and update
    scaler.step(optimizer)
    scaler.update()
</syntaxhighlight>

'''Loss Scaling (FP16):'''
Small gradients underflow to zero in FP16. Solution:
1. Scale loss by large factor (e.g., 1024)
2. Gradients are scaled up, avoiding underflow
3. Unscale before optimizer step

BF16 doesn't need loss scaling due to larger exponent range.

'''Memory Savings:'''
{| class="wikitable"
! Component !! FP32 !! Mixed Precision
|-
|| Model weights || 4 bytes/param || 2 bytes/param
|-
|| Activations || 4 bytes || 2 bytes
|-
|| Gradients || 4 bytes/param || 2 bytes/param
|-
|| Master weights || 4 bytes/param || 4 bytes/param (kept FP32)
|}

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:Unsloth_FastLanguageModel]]

=== Tips and Tricks ===
(Handled automatically by Unsloth)

