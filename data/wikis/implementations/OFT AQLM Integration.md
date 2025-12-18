= OFT AQLM Integration =

== Knowledge Sources ==

* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* [https://arxiv.org/abs/2306.07280 OFT: Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* [https://arxiv.org/abs/2306.12929 AQLM: Extreme Compression of Large Language Models via Additive Quantization]

== Domains ==

[[Category:NLP]]
[[Category:PEFT]]
[[Category:Orthogonal_Fine_Tuning]]
[[Category:Quantization]]
[[Category:AQLM]]

== Overview ==

=== Description ===

The <code>AqlmOFTLinear</code> class provides an implementation of Orthogonal Fine-Tuning (OFT) for AQLM (Additive Quantization for Large Models) quantized linear layers. This integration enables parameter-efficient fine-tuning of models quantized with AQLM, combining the benefits of both orthogonal fine-tuning and extreme model compression.

AQLM is a quantization method that uses additive quantization techniques to achieve extreme compression of large language models. The OFT adapter wraps AQLM's <code>QuantizedLinear</code> layers, allowing fine-tuning while maintaining the quantized model structure.

Key features:
* Supports AQLM quantized models with OFT adapters
* Preserves quantization during fine-tuning
* Applies orthogonal transformations before quantized computation
* Handles dtype conversions appropriately for autocast scenarios

=== Usage ===

This module is used internally by the PEFT library when applying OFT to AQLM-quantized models. The dispatcher function <code>dispatch_aqlm</code> automatically detects AQLM quantized layers and wraps them with the appropriate OFT adapter.

== Code Reference ==

=== Source Location ===

File: <code>src/peft/tuners/oft/aqlm.py</code>

Repository: HuggingFace PEFT (Parameter-Efficient Fine-Tuning)

=== Class: AqlmOFTLinear ===

==== Signature ====

<syntaxhighlight lang="python">
class AqlmOFTLinear(torch.nn.Module, OFTLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        oft_block_size: int = 32,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        fan_in_fan_out: bool = False,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        **kwargs,
    )
</syntaxhighlight>

==== Import ====

<syntaxhighlight lang="python">
from peft.tuners.oft.aqlm import AqlmOFTLinear, dispatch_aqlm
</syntaxhighlight>

== I/O Contract ==

=== Constructor Parameters ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| base_layer || Module || required || The AQLM quantized linear layer to wrap
|-
| adapter_name || str || required || Name identifier for this adapter
|-
| r || int || 0 || OFT rank (number of OFT blocks per injected layer)
|-
| oft_block_size || int || 32 || Size of OFT blocks across different layers
|-
| module_dropout || float || 0.0 || Multiplicative dropout probability for OFT blocks
|-
| init_weights || bool || True || Whether to initialize OFT weights
|-
| coft || bool || False || Whether to use constrained OFT variant
|-
| eps || float || 6e-5 || Control strength for COFT (only used if coft=True)
|-
| block_share || bool || False || Whether to share OFT parameters between blocks
|-
| fan_in_fan_out || bool || False || Set to True if layer stores weights as (fan_in, fan_out)
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

=== dispatch_aqlm Function ===

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
| new_module || Optional[torch.nn.Module] || AqlmOFTLinear instance if target is AQLM quantized, None otherwise
|}

== Usage Examples ==

=== Basic Usage with PEFT ===

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import get_peft_model, OFTConfig

# Load a model quantized with AQLM
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config={"method": "aqlm"}
)

# Configure OFT
oft_config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    module_dropout=0.1,
    coft=False,
)

# Apply OFT - automatically uses AqlmOFTLinear for AQLM layers
peft_model = get_peft_model(model, oft_config)

# Fine-tune the model
# ... training code ...
</syntaxhighlight>

=== Explicit Dispatcher Usage ===

<syntaxhighlight lang="python">
from peft.tuners.oft.aqlm import dispatch_aqlm
from aqlm import QuantizedLinear

# Assume we have an AQLM quantized layer
aqlm_layer = QuantizedLinear(...)

# Dispatch to create OFT adapter
oft_layer = dispatch_aqlm(
    target=aqlm_layer,
    adapter_name="default",
    r=8,
    oft_block_size=32,
    module_dropout=0.1
)

# Use the OFT-wrapped layer
output = oft_layer(input_tensor)
</syntaxhighlight>

=== Advanced Configuration ===

<syntaxhighlight lang="python">
from peft import OFTConfig

# Configure OFT with advanced options
oft_config = OFTConfig(
    r=16,
    oft_block_size=0,  # Will be computed from r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    module_dropout=0.05,
    coft=True,  # Use constrained OFT
    eps=1e-4,
    block_share=False,
    use_cayley_neumann=True,
    num_cayley_neumann_terms=7,
)

# Apply to AQLM quantized model
peft_model = get_peft_model(aqlm_quantized_model, oft_config)
</syntaxhighlight>

== Implementation Details ==

=== Forward Pass Logic ===

The forward method implements the following logic:

1. Check if adapters are disabled - if so, bypass OFT and use base layer directly
2. For each active adapter:
   * Apply dtype conversion if not in autocast mode
   * Apply the OFT rotation transformation (oft_R)
3. Pass transformed input through the AQLM quantized base layer
4. Convert result back to expected dtype if needed

=== Quantization Weight Handling ===

The dispatcher updates the <code>qweight</code> attribute to reference the AQLM quantized weights:

<syntaxhighlight lang="python">
target.qweight = target_base_layer.codes
</syntaxhighlight>

This ensures proper access to the quantized weight codes used by AQLM.

=== Merging Not Supported ===

Unlike some other OFT implementations, merging OFT weights into AQLM quantized weights is not supported due to the complexity of the quantization scheme.

== Related Pages ==

* [[huggingface_peft_OFTConfig|OFTConfig]] - Configuration class for OFT
* [[huggingface_peft_OFTModel|OFTModel]] - Main OFT model implementation
* [[huggingface_peft_OFT_AWQ|OFT AWQ Integration]] - Similar integration for AWQ quantization
* [[huggingface_peft_OFT_GPTQ|OFT GPTQ Integration]] - Similar integration for GPTQ quantization
* [[huggingface_peft_OFT_HQQ|OFT HQQ Integration]] - Similar integration for HQQ quantization

== See Also ==

* OFT Paper: [https://arxiv.org/abs/2306.07280 Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* AQLM Paper: [https://arxiv.org/abs/2306.12929 Extreme Compression via Additive Quantization]
* PEFT Documentation: [https://huggingface.co/docs/peft HuggingFace PEFT Docs]
