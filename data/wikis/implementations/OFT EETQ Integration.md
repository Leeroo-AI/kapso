= OFT EETQ Integration =

== Knowledge Sources ==

* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* [https://arxiv.org/abs/2306.07280 OFT: Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* [https://github.com/NetEase-FuXi/EETQ EETQ: Easy & Efficient Quantization for Transformers]

== Domains ==

[[Category:NLP]]
[[Category:PEFT]]
[[Category:Orthogonal_Fine_Tuning]]
[[Category:Quantization]]
[[Category:EETQ]]

== Overview ==

=== Description ===

The <code>EetqOFTLinear</code> class provides an implementation of Orthogonal Fine-Tuning (OFT) for EETQ (Easy & Efficient Quantization for Transformers) quantized linear layers. This integration enables parameter-efficient fine-tuning of models quantized with EETQ, combining efficient quantization with orthogonal adaptation.

EETQ is a quantization method designed for ease of use and computational efficiency in transformer models. The OFT adapter wraps EETQ's <code>EetqLinear</code> layers, allowing fine-tuning while maintaining the quantized model structure.

Key features:
* Supports EETQ quantized models with OFT adapters
* Preserves quantization during fine-tuning
* Applies orthogonal transformations before quantized computation
* Handles dtype conversions appropriately for autocast scenarios
* Explicitly disables merging/unmerging operations (not supported for EETQ)

=== Usage ===

This module is used internally by the PEFT library when applying OFT to EETQ-quantized models. The dispatcher function <code>dispatch_eetq</code> automatically detects EETQ quantized layers and wraps them with the appropriate OFT adapter.

== Code Reference ==

=== Source Location ===

File: <code>src/peft/tuners/oft/eetq.py</code>

Repository: HuggingFace PEFT (Parameter-Efficient Fine-Tuning)

=== Class: EetqOFTLinear ===

==== Signature ====

<syntaxhighlight lang="python">
class EetqOFTLinear(torch.nn.Module, OFTLayer):
    def __init__(
        self,
        base_layer,
        adapter_name,
        r: int = 0,
        oft_block_size: int = 0,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        fan_in_fan_out: bool = False,
        **kwargs,
    )
</syntaxhighlight>

==== Import ====

<syntaxhighlight lang="python">
from peft.tuners.oft.eetq import EetqOFTLinear, dispatch_eetq
</syntaxhighlight>

== I/O Contract ==

=== Constructor Parameters ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| base_layer || Module || required || The EETQ quantized linear layer to wrap
|-
| adapter_name || str || required || Name identifier for this adapter
|-
| r || int || 0 || OFT rank (number of OFT blocks per injected layer)
|-
| oft_block_size || int || 0 || Size of OFT blocks across different layers
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
| use_cayley_neumann || bool || False || Whether to use Cayley-Neumann formulation
|-
| num_cayley_neumann_terms || int || 5 || Number of terms in Cayley-Neumann approximation
|-
| fan_in_fan_out || bool || False || Set to True if layer stores weights as (fan_in, fan_out)
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

=== dispatch_eetq Function ===

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
| new_module || Optional[torch.nn.Module] || EetqOFTLinear instance if target is EETQ quantized, None otherwise
|}

== Usage Examples ==

=== Basic Usage with PEFT ===

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import get_peft_model, OFTConfig

# Load a model quantized with EETQ
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config={"method": "eetq"}
)

# Configure OFT
oft_config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    module_dropout=0.1,
    coft=False,
)

# Apply OFT - automatically uses EetqOFTLinear for EETQ layers
peft_model = get_peft_model(model, oft_config)

# Fine-tune the model
# ... training code ...
</syntaxhighlight>

=== Advanced Configuration ===

<syntaxhighlight lang="python">
from peft import OFTConfig, get_peft_model

# Configure OFT with advanced options
oft_config = OFTConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    module_dropout=0.05,
    coft=True,  # Use constrained OFT
    eps=1e-4,
    block_share=False,
    use_cayley_neumann=True,
    num_cayley_neumann_terms=7,
)

# Apply to EETQ quantized model
peft_model = get_peft_model(eetq_quantized_model, oft_config)
</syntaxhighlight>

=== Explicit Dispatcher Usage ===

<syntaxhighlight lang="python">
from peft.tuners.oft.eetq import dispatch_eetq
from eetq import EetqLinear

# Assume we have an EETQ quantized layer
eetq_layer = EetqLinear(...)

# Dispatch to create OFT adapter
oft_layer = dispatch_eetq(
    target=eetq_layer,
    adapter_name="default",
    r=8,
    oft_block_size=0,
    module_dropout=0.1
)

# Use the OFT-wrapped layer
output = oft_layer(input_tensor)
</syntaxhighlight>

=== Handling Merge Operations ===

<syntaxhighlight lang="python">
# Note: Merging is not supported for EETQ layers
try:
    peft_model.merge_adapter()
except AttributeError as e:
    print(e)  # "Merging LoRA layers is not supported for Eetq layers."

# Similarly for unmerge
try:
    peft_model.unmerge_adapter()
except AttributeError as e:
    print(e)  # "Unmerging LoRA layers is not supported for Eetq layers."
</syntaxhighlight>

== Implementation Details ==

=== Forward Pass Logic ===

The forward method implements the following logic:

1. Check if adapters are disabled - if so, bypass OFT and use base layer directly
2. For each active adapter:
   * Apply dtype conversion if not in autocast mode
   * Apply the OFT rotation transformation (oft_R)
3. Pass transformed input through the EETQ quantized base layer
4. Convert result back to expected dtype if needed

=== Merge/Unmerge Not Supported ===

Unlike some other OFT implementations, EETQ does not support merging adapter weights into the base layer:

<syntaxhighlight lang="python">
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    raise AttributeError("Merging LoRA layers is not supported for Eetq layers.")

def unmerge(self) -> None:
    raise AttributeError("Unmerging LoRA layers is not supported for Eetq layers.")
</syntaxhighlight>

This is because the EETQ quantization format does not allow for easy weight merging without dequantization.

=== Weight and Bias Handling ===

The dispatcher copies weight and bias references from the base layer:

<syntaxhighlight lang="python">
target.weight = target_base_layer.weight

if hasattr(target, "bias"):
    target.bias = target_base_layer.bias
</syntaxhighlight>

=== Backward Compatibility ===

The class maintains both <code>base_layer</code> and <code>quant_linear_module</code> attributes pointing to the same object:
* <code>base_layer</code> - for consistency with other OFT implementations
* <code>quant_linear_module</code> - for backward compatibility with older code

== Related Pages ==

* [[huggingface_peft_OFTConfig|OFTConfig]] - Configuration class for OFT
* [[huggingface_peft_OFTModel|OFTModel]] - Main OFT model implementation
* [[huggingface_peft_OFT_AQLM|OFT AQLM Integration]] - Similar integration for AQLM quantization
* [[huggingface_peft_OFT_AWQ|OFT AWQ Integration]] - Similar integration for AWQ quantization
* [[huggingface_peft_OFT_GPTQ|OFT GPTQ Integration]] - Similar integration for GPTQ quantization
* [[huggingface_peft_OFT_HQQ|OFT HQQ Integration]] - Similar integration for HQQ quantization

== See Also ==

* OFT Paper: [https://arxiv.org/abs/2306.07280 Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* EETQ Repository: [https://github.com/NetEase-FuXi/EETQ EETQ on GitHub]
* PEFT Documentation: [https://huggingface.co/docs/peft HuggingFace PEFT Docs]
