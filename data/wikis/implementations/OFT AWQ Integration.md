= OFT AWQ Integration =

== Knowledge Sources ==

* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* [https://arxiv.org/abs/2306.07280 OFT: Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* [https://arxiv.org/abs/2306.00978 AWQ: Activation-aware Weight Quantization]

== Domains ==

[[Category:NLP]]
[[Category:PEFT]]
[[Category:Orthogonal_Fine_Tuning]]
[[Category:Quantization]]
[[Category:AWQ]]

== Overview ==

=== Description ===

The <code>AwqOFTLinear</code> class provides an implementation of Orthogonal Fine-Tuning (OFT) for AWQ (Activation-aware Weight Quantization) quantized linear layers. This integration enables parameter-efficient fine-tuning of models quantized with AWQ, allowing practitioners to fine-tune compressed models while maintaining their memory efficiency.

AWQ is a quantization method that uses activation-aware techniques to achieve high-quality quantization with minimal accuracy loss. The OFT adapter wraps AWQ's <code>WQLinear_GEMM</code> layers, allowing fine-tuning while preserving the quantized structure.

Key features:
* Supports AWQ quantized models with OFT adapters
* Version checking to ensure compatibility (requires autoawq >= 0.2.0)
* Preserves quantization during fine-tuning
* Applies orthogonal transformations before quantized computation
* Handles dtype conversions for autocast scenarios

=== Usage ===

This module is used internally by the PEFT library when applying OFT to AWQ-quantized models. The dispatcher function <code>dispatch_awq</code> automatically detects AWQ quantized layers and wraps them with the appropriate OFT adapter.

== Code Reference ==

=== Source Location ===

File: <code>src/peft/tuners/oft/awq.py</code>

Repository: HuggingFace PEFT (Parameter-Efficient Fine-Tuning)

=== Class: AwqOFTLinear ===

==== Signature ====

<syntaxhighlight lang="python">
class AwqOFTLinear(torch.nn.Module, OFTLayer):
    def __init__(
        self,
        base_layer,
        adapter_name,
        r: int = 0,
        oft_block_size: int = 32,
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        fan_in_fan_out: bool = False,
        init_weights: bool = True,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        **kwargs,
    )
</syntaxhighlight>

==== Import ====

<syntaxhighlight lang="python">
from peft.tuners.oft.awq import AwqOFTLinear, dispatch_awq
</syntaxhighlight>

== I/O Contract ==

=== Constructor Parameters ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| base_layer || Module || required || The AWQ quantized linear layer to wrap
|-
| adapter_name || str || required || Name identifier for this adapter
|-
| r || int || 0 || OFT rank (number of OFT blocks per injected layer)
|-
| oft_block_size || int || 32 || Size of OFT blocks across different layers
|-
| module_dropout || float || 0.0 || Multiplicative dropout probability for OFT blocks
|-
| coft || bool || False || Whether to use constrained OFT variant
|-
| eps || float || 6e-5 || Control strength for COFT (only used if coft=True)
|-
| block_share || bool || False || Whether to share OFT parameters between blocks
|-
| fan_in_fan_out || bool || False || Set to True if layer stores weights as (fan_in, fan_out)
|-
| init_weights || bool || True || Whether to initialize OFT weights
|-
| use_cayley_neumann || bool || False || Whether to use Cayley-Neumann formulation
|-
| num_cayley_neumann_terms || int || 5 || Number of terms in Cayley-Neumann approximation
|}

=== Forward Method ===

{| class="wikitable"
! Input !! Type !! Description
|-
| x || torch.Tensor || Input tensor to the layer
|}

{| class="wikitable"
! Output !! Type !! Description
|-
| result || torch.Tensor || Output tensor after applying OFT transformation and quantized computation
|}

=== dispatch_awq Function ===

{| class="wikitable"
! Parameter !! Type !! Description
|-
| target || torch.nn.Module || The target module to potentially wrap
|-
| adapter_name || str || Name for the adapter
|-
| **kwargs || Any || Additional keyword arguments passed to constructor
|}

{| class="wikitable"
! Return !! Type !! Description
|-
| new_module || Optional[torch.nn.Module] || AwqOFTLinear instance if target is AWQ quantized, None otherwise
|}

== Usage Examples ==

=== Basic Usage with PEFT ===

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import get_peft_model, OFTConfig

# Load a model quantized with AWQ
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config={"method": "awq"}
)

# Configure OFT
oft_config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    module_dropout=0.1,
    coft=False,
)

# Apply OFT - automatically uses AwqOFTLinear for AWQ layers
peft_model = get_peft_model(model, oft_config)

# Fine-tune the model
# ... training code ...
</syntaxhighlight>

=== Version Compatibility Check ===

<syntaxhighlight lang="python">
import importlib.metadata as importlib_metadata
import packaging.version

# The dispatcher automatically checks version compatibility
# Minimum required version is 0.2.0
AUTOAWQ_MINIMUM_VERSION = packaging.version.parse("0.2.0")
version_autoawq = packaging.version.parse(
    importlib_metadata.version("autoawq")
)

if AUTOAWQ_MINIMUM_VERSION > version_autoawq:
    raise ImportError(
        f"Found incompatible version {version_autoawq}, "
        f"but only versions above {AUTOAWQ_MINIMUM_VERSION} are supported"
    )
</syntaxhighlight>

=== Advanced Configuration with Constrained OFT ===

<syntaxhighlight lang="python">
from peft import OFTConfig, get_peft_model

# Configure OFT with constrained variant
oft_config = OFTConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    module_dropout=0.05,
    coft=True,  # Use constrained OFT
    eps=1e-4,   # Control freedom of rotation
    block_share=False,
    use_cayley_neumann=True,
    num_cayley_neumann_terms=7,
)

# Apply to AWQ quantized model
peft_model = get_peft_model(awq_quantized_model, oft_config)
</syntaxhighlight>

=== Explicit Dispatcher Usage ===

<syntaxhighlight lang="python">
from peft.tuners.oft.awq import dispatch_awq
from awq.modules.linear import WQLinear_GEMM

# Assume we have an AWQ quantized layer
awq_layer = WQLinear_GEMM(...)

# Dispatch to create OFT adapter
oft_layer = dispatch_awq(
    target=awq_layer,
    adapter_name="default",
    r=8,
    oft_block_size=32,
    module_dropout=0.1
)

# Use the OFT-wrapped layer
output = oft_layer(input_tensor)
</syntaxhighlight>

== Implementation Details ==

=== Forward Pass Logic ===

The forward method implements the following logic:

1. Check if adapters are disabled - if so, bypass OFT and use base layer directly
2. For each active adapter:
   * Apply dtype conversion if not in autocast mode
   * Apply the OFT rotation transformation (oft_R)
   * Convert back to expected dtype if needed
3. Pass transformed input through the AWQ quantized base layer

=== Quantization Weight Handling ===

The dispatcher updates the <code>qweight</code> attribute to reference the AWQ quantized weights:

<syntaxhighlight lang="python">
target.qweight = target_base_layer.qweight
</syntaxhighlight>

This ensures proper access to the quantized weights used by AWQ.

=== Backward Compatibility ===

The class maintains both <code>base_layer</code> and <code>quant_linear_module</code> attributes pointing to the same object:
* <code>base_layer</code> - for consistency with other OFT implementations
* <code>quant_linear_module</code> - for backward compatibility with older code

=== Version Requirements ===

The implementation requires <code>autoawq >= 0.2.0</code>. The dispatcher automatically checks this requirement and raises an informative error if an incompatible version is detected.

== Related Pages ==

* [[huggingface_peft_OFTConfig|OFTConfig]] - Configuration class for OFT
* [[huggingface_peft_OFTModel|OFTModel]] - Main OFT model implementation
* [[huggingface_peft_OFT_AQLM|OFT AQLM Integration]] - Similar integration for AQLM quantization
* [[huggingface_peft_OFT_GPTQ|OFT GPTQ Integration]] - Similar integration for GPTQ quantization
* [[huggingface_peft_OFT_EETQ|OFT EETQ Integration]] - Similar integration for EETQ quantization

== See Also ==

* OFT Paper: [https://arxiv.org/abs/2306.07280 Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* AWQ Paper: [https://arxiv.org/abs/2306.00978 Activation-aware Weight Quantization]
* PEFT Documentation: [https://huggingface.co/docs/peft HuggingFace PEFT Docs]
