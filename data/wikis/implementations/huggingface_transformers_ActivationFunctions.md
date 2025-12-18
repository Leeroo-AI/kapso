{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Neural_Networks]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A comprehensive collection of activation function implementations for neural networks, including multiple GELU variants, SiLU, Mish, and the experimental xIELU activation with optional CUDA acceleration.

=== Description ===
This module provides PyTorch nn.Module implementations of various activation functions used throughout the Transformers library. It includes multiple GELU (Gaussian Error Linear Unit) approximations optimized for different speed/accuracy tradeoffs: GELUTanh (fast C implementation), NewGELUActivation (tanh approximation), GELUActivation (original erf-based), FastGELUActivation, and QuickGELUActivation.

The module also implements specialized activations like ClippedGELU (for quantization), AccurateGELU, MishActivation, LinearActivation, LaplaceActivation (for MEGA attention), ReLUSquaredActivation, and the advanced XIELUActivation with learnable parameters. Many functions are decorated with @use_kernel_forward_from_hub to allow swapping implementations from the Hugging Face Hub.

The ACT2FN dictionary provides string-based lookup using ClassInstantier, enabling configuration-driven activation selection. The get_activation() function retrieves activation modules by name. For backwards compatibility, standalone function references (gelu, gelu_new, silu, mish, etc.) are provided at module level.

=== Usage ===
Activation functions are used in model configuration files and within model architectures. Models typically specify activation type as a string (e.g., "gelu", "silu") in their config, which is resolved via get_activation(). The functions can also be instantiated directly as nn.Module layers within custom architectures.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' src/transformers/activations.py

=== Signature ===
<syntaxhighlight lang="python">
# Main lookup function
def get_activation(activation_string: str) -> nn.Module:
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found...")

# Key activation classes
class GELUTanh(nn.Module):
    def __init__(self, use_gelu_tanh_python: bool = False): ...
    def forward(self, input: Tensor) -> Tensor: ...

class XIELUActivation(nn.Module):
    def __init__(self, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5,
                 eps=-1e-6, dtype=torch.bfloat16, with_vector_loads=False): ...
    def forward(self, input: Tensor) -> Tensor: ...

class ClippedGELUActivation(nn.Module):
    def __init__(self, min: float, max: float): ...
    def forward(self, x: Tensor) -> Tensor: ...

# Dictionary mapping
ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_fast": FastGELUActivation,
    "silu": SiLUActivation,
    "mish": MishActivation,
    "xielu": XIELUActivation,
    # ... 20+ activation types
}
ACT2FN = ClassInstantier(ACT2CLS)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Import via function lookup (recommended)
from transformers.activations import get_activation
act_fn = get_activation("gelu_fast")

# Import specific activation classes
from transformers.activations import GELUActivation, SiLUActivation, XIELUActivation

# Import dictionary for configuration
from transformers.activations import ACT2FN

# Backwards compatibility imports
from transformers.activations import gelu, gelu_new, silu, mish
</syntaxhighlight>

== I/O Contract ==

=== Available Activation Functions ===
{| class="wikitable"
|-
! String Key !! Class !! Properties !! Use Case
|-
| "gelu" || GELUActivation || Original erf-based GELU || High accuracy requirements
|-
| "gelu_fast" || FastGELUActivation || Tanh approximation (0.7978845608 constant) || Good speed/accuracy balance
|-
| "gelu_new" || NewGELUActivation || Tanh approximation (BERT/GPT style) || BERT-compatible models
|-
| "gelu_pytorch_tanh" || GELUTanh || Fast C implementation via torch.functional || Fastest GELU variant
|-
| "quick_gelu" || QuickGELUActivation || Sigmoid approximation (1.702 * x) || Fastest but less accurate
|-
| "gelu_accurate" || AccurateGELUActivation || Precomputed constants || MEGA architecture
|-
| "gelu_10" || ClippedGELUActivation || Clipped to [-10, 10] || Quantization-friendly
|-
| "silu" / "swish" || SiLUActivation / nn.SiLU || x * sigmoid(x) || Modern architectures
|-
| "mish" || MishActivation || x * tanh(softplus(x)) || Self-regularized activation
|-
| "relu" || nn.ReLU || Standard ReLU || Baseline activation
|-
| "relu2" || ReLUSquaredActivation || ReLU(x)^2 || Primer paper
|-
| "relu6" || nn.ReLU6 || min(max(0, x), 6) || Mobile models
|-
| "leaky_relu" || nn.LeakyReLU || Negative slope variant || Avoiding dead neurons
|-
| "prelu" || nn.PReLU || Learnable negative slope || Adaptive activation
|-
| "laplace" || LaplaceActivation || Laplace CDF-based || MEGA attention
|-
| "linear" || LinearActivation || Identity function || No activation
|-
| "xielu" || XIELUActivation || Learnable quadratic (experimental) || Advanced research
|}

=== XIELUActivation Parameters ===
{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| alpha_p_init || float || 0.8 || Initial positive region slope
|-
| alpha_n_init || float || 0.8 || Initial negative region slope
|-
| beta || float || 0.5 || Linear component weight
|-
| eps || float || -1e-6 || Numerical stability epsilon
|-
| dtype || torch.dtype || torch.bfloat16 || Parameter dtype
|-
| with_vector_loads || bool || False || CUDA kernel optimization flag
|}

=== Input/Output ===
{| class="wikitable"
|-
! Method !! Input !! Output
|-
| forward() || Tensor of any shape || Tensor of same shape with activation applied
|-
| get_activation() || str (activation name) || nn.Module instance
|}

== Usage Examples ==
<syntaxhighlight lang="python">
import torch
from transformers.activations import get_activation, ACT2FN, XIELUActivation

# Using get_activation (recommended for config-based models)
act_fn = get_activation("gelu_fast")
x = torch.randn(2, 512)
output = act_fn(x)
print(output.shape)  # torch.Size([2, 512])

# Direct class instantiation
from transformers.activations import GELUTanh
gelu = GELUTanh()
y = gelu(torch.randn(10, 768))

# Using with model configuration
model_config = {"hidden_act": "silu"}
activation = ACT2FN[model_config["hidden_act"]]

# Clipped GELU for quantization
clipped_gelu = get_activation("gelu_10")  # Clips to [-10, 10]

# Advanced: xIELU with custom parameters
xielu = XIELUActivation(
    alpha_p_init=0.9,
    alpha_n_init=0.7,
    beta=0.3,
    dtype=torch.float32
)
z = xielu(torch.randn(4, 256))

# In custom nn.Module
class CustomLayer(torch.nn.Module):
    def __init__(self, hidden_size, activation="gelu"):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.dense(x))

layer = CustomLayer(768, activation="silu")

# Performance comparison
import time
x = torch.randn(1000, 4096).cuda()

for act_name in ["gelu", "gelu_fast", "quick_gelu"]:
    act = get_activation(act_name).cuda()
    start = time.time()
    for _ in range(100):
        _ = act(x)
    print(f"{act_name}: {(time.time() - start) * 1000:.2f}ms")

# Backwards compatibility
from transformers.activations import gelu, silu, mish
legacy_output = gelu(torch.randn(10, 10))
</syntaxhighlight>

== Related Pages ==
* (Leave empty)
