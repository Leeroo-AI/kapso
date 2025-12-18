{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Fourier_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Layer implementation for FourierFT that learns sparse spectral components and applies inverse FFT to compute weight deltas.

=== Description ===

FourierFTLayer and FourierFTLinear implement frequency-domain adaptation. The trainable parameter fourierft_spectrum has n_frequency elements representing sparse Fourier coefficients. Random indices (controlled by random_loc_seed) select which spectral entries are learned. Delta weight is computed via inverse FFT: ifft2(sparse_spectrum) * scaling.

=== Usage ===

Use FourierFT layers for extreme parameter efficiency. The sparse spectrum approach requires far fewer parameters than LoRA for similar quality. Supports merge/unmerge and multi-adapter inference.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/fourierft/layer.py src/peft/tuners/fourierft/layer.py]
* '''Lines:''' 1-194

=== Signature ===
<syntaxhighlight lang="python">
class FourierFTLayer(BaseTunerLayer):
    """Base FourierFT layer class."""
    adapter_layer_names = ("fourierft_spectrum",)

    def update_layer(self, adapter_name, n_frequency, scaling, init_weights, ...):
        """Create spectrum parameter and random indices."""

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """Compute delta via ifft2(sparse_spectrum) * scaling."""

class FourierFTLinear(nn.Module, FourierFTLayer):
    """FourierFT implementation for Linear layers."""
    def forward(self, x: torch.Tensor, ...) -> torch.Tensor:
        """Apply Fourier-domain adaptation."""

    def merge(self, safe_merge: bool = False, adapter_names=None):
        """Merge delta weight into base weights."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.fourierft import FourierFTLayer, FourierFTLinear
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
| n_frequency || int || No || Number of spectral components
|-
| scaling || float || No || Output scaling factor
|-
| random_loc_seed || int || No || Seed for index selection
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Base output + Fourier adaptation
|-
| get_delta_weight() || torch.Tensor || ifft2(sparse_spectrum) * scaling
|}

== Usage Examples ==

=== FourierFT Forward Pass ===
<syntaxhighlight lang="python">
# FourierFT forward computation:
# 1. Get base layer output
result = self.base_layer(x)

# 2. Compute delta weight from sparse spectrum
delta_w = self.get_delta_weight(adapter)
# delta_w = ifft2(sparse_spectrum) * scaling

# 3. Add Fourier adaptation
result = result + F.linear(x, delta_w)
</syntaxhighlight>

=== Delta Weight Computation ===
<syntaxhighlight lang="python">
def get_delta_weight(self, adapter):
    spectrum = self.fourierft_spectrum[adapter]  # [n_frequency]
    indices = self.indices[adapter]  # Random spectral locations

    # Create sparse spectrum
    dense_spectrum = torch.zeros(out_features, in_features)
    dense_spectrum[indices[0], indices[1]] = spectrum

    # Inverse FFT to get delta weight
    delta_weight = torch.fft.ifft2(dense_spectrum).real
    return delta_weight * self.fourierft_scaling[adapter]
</syntaxhighlight>

=== Initialization ===
<syntaxhighlight lang="python">
# init_weights=True: zeros (no-op)
# init_weights=False: standard normal (default)

# Random indices are deterministic based on random_loc_seed
# Same seed = same spectral locations = reproducible
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
