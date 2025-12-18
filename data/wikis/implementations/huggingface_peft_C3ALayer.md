{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Circulant_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Layer implementation for C3A that applies block circulant convolution via FFT for parameter-efficient adaptation.

=== Description ===

C3ALayer and C3ALinear implement circulant convolution adaptation. The trainable parameter c3a_kernel has shape [out_features/block_size, in_features/block_size, block_size]. Forward pass applies BlockCircularConvolution (via FFT) and divides by input size. Delta weight is computed via get_circulant_fast which constructs the full circulant matrix. FFT operations use float32.

=== Usage ===

Use C3A layers for FFT-based circulant adaptation. Layers support merge/unmerge and multi-adapter inference. Block size must divide both input and output dimensions.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/c3a/layer.py src/peft/tuners/c3a/layer.py]
* '''Lines:''' 1-203

=== Signature ===
<syntaxhighlight lang="python">
class C3ALayer(BaseTunerLayer):
    """Base C3A layer class."""
    adapter_layer_names = ("c3a_kernel",)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """Get circulant matrix from kernel."""

    def update_layer(self, adapter_name, block_size, init_weights, ...):
        """Create c3a_kernel parameter."""

class C3ALinear(nn.Module, C3ALayer):
    """C3A implementation for Linear layers."""
    def forward(self, x: torch.Tensor, ...) -> torch.Tensor:
        """Apply block circulant convolution."""

    def merge(self, safe_merge: bool = False, adapter_names=None):
        """Merge circulant matrix into weights."""

    def unmerge(self):
        """Remove merged adaptation."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.c3a import C3ALayer, C3ALinear
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Linear || Yes || Base linear layer
|-
| adapter_name || str || Yes || Adapter name
|-
| block_size || int || Yes || Circulant block size
|-
| init_weights || str/bool || Yes || Initialization method
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Base output + circulant convolution
|-
| get_delta_weight() || torch.Tensor || Full circulant matrix
|}

== Usage Examples ==

=== C3A Forward Pass ===
<syntaxhighlight lang="python">
# C3A forward computation:
# 1. Get base layer output
result = self.base_layer(x)

# 2. Apply block circulant convolution (in float32)
x = x.to(torch.float32)
for adapter in self.active_adapters:
    kernel = self.c3a_kernel[adapter]
    x = BlockCircularConvolution.apply(x, kernel) / x.size(-1)
    result += x.to(result.dtype)
</syntaxhighlight>

=== Kernel Shape ===
<syntaxhighlight lang="python">
# c3a_kernel shape:
# [out_features // block_size, in_features // block_size, block_size]

# Example: Linear(4096, 4096) with block_size=256
# kernel shape: [16, 16, 256]
# Total params: 16 * 16 * 256 = 65,536
</syntaxhighlight>

=== Initialization Options ===
<syntaxhighlight lang="python">
# True: zeros (no-op initialization)
# "xavier_uniform": Xavier uniform (default)
# "kaiming_uniform": Kaiming uniform
# "gaussian": Normal distribution
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
