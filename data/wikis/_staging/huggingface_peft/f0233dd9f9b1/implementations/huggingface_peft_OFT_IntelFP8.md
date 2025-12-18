= OFT Intel Neural Compressor FP8 Integration =

== Knowledge Sources ==

* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* [https://arxiv.org/abs/2306.07280 OFT: Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* [https://github.com/intel/neural-compressor Intel Neural Compressor]
* [https://github.com/huggingface/optimum-habana Optimum Habana]

== Domains ==

[[Category:NLP]]
[[Category:PEFT]]
[[Category:Orthogonal_Fine_Tuning]]
[[Category:Quantization]]
[[Category:Intel]]
[[Category:FP8]]
[[Category:Habana]]

== Overview ==

=== Description ===

The <code>IncOFTLinear</code> class provides an implementation of Orthogonal Fine-Tuning (OFT) for Intel Neural Compressor (INC) FP8 quantized linear layers. This integration enables parameter-efficient fine-tuning of models quantized with Intel's Neural Compressor for FP8 precision, particularly for use with Intel Habana Gaudi accelerators.

Intel Neural Compressor is a quantization library optimized for Intel hardware, including Habana Gaudi AI accelerators. The FP8 (8-bit floating point) quantization provides a balance between model size, computational efficiency, and accuracy. The OFT adapter wraps INC's <code>PatchedLinear</code> layers, allowing fine-tuning while maintaining the quantized structure.

Key features:
* Supports Intel Neural Compressor FP8 quantized models with OFT adapters
* Optimized for Intel Habana Gaudi hardware
* Extends the standard Linear OFT implementation
* Note: Merge/unmerge operations not yet implemented
* Tests are handled in the Optimum-Habana repository

=== Usage ===

This module is used internally by the PEFT library when applying OFT to INC FP8-quantized models. The dispatcher function <code>dispatch_inc</code> automatically detects INC quantized layers and wraps them with the appropriate OFT adapter.

== Code Reference ==

=== Source Location ===

File: <code>src/peft/tuners/oft/inc.py</code>

Repository: HuggingFace PEFT (Parameter-Efficient Fine-Tuning)

=== Class: IncOFTLinear ===

==== Signature ====

<syntaxhighlight lang="python">
class IncOFTLinear(Linear):
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        **kwargs,
    )
</syntaxhighlight>

==== Import ====

<syntaxhighlight lang="python">
from peft.tuners.oft.inc import IncOFTLinear, dispatch_inc
</syntaxhighlight>

== I/O Contract ==

=== Constructor Parameters ===

{| class="wikitable"
! Parameter !! Type !! Description
|-
| base_layer || torch.nn.Module || The INC FP8 quantized linear layer (PatchedLinear) to wrap
|-
| adapter_name || str || Name identifier for this adapter
|-
| **kwargs || Any || Additional keyword arguments passed to parent Linear class
|}

The <code>**kwargs</code> are passed to the parent <code>Linear</code> class and include all standard OFT parameters:
* <code>r</code>: OFT rank
* <code>oft_block_size</code>: Block size
* <code>module_dropout</code>: Dropout probability
* <code>coft</code>: Constrained OFT flag
* <code>eps</code>: COFT epsilon
* <code>block_share</code>: Block sharing flag
* <code>use_cayley_neumann</code>: Cayley-Neumann flag
* <code>num_cayley_neumann_terms</code>: Number of Cayley-Neumann terms
* And other parameters from the Linear parent class

=== Forward Method ===

Inherited from the parent <code>Linear</code> class.

{| class="wikitable"
! Input !! Type !! Description
|-
| x || torch.Tensor || Input tensor to the layer
|-
| *args || Any || Additional positional arguments
|-
| **kwargs || Any || Additional keyword arguments
|}

{| class="wikitable"
! Output !! Type !! Description
|-
| result || torch.Tensor || Output tensor after applying OFT transformation and quantized computation
|}

=== Merge/Unmerge Methods ===

Currently raise <code>NotImplementedError</code>:

{| class="wikitable"
! Method !! Status !! Error Message
|-
| merge() || Not Implemented || "Merging OFT with INC layers is not yet implemented"
|-
| unmerge() || Not Implemented || "Unmerging OFT from INC layers is not yet implemented"
|}

=== dispatch_inc Function ===

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
| new_module || Optional[torch.nn.Module] || IncOFTLinear instance if target is INC quantized, None otherwise
|}

== Usage Examples ==

=== Basic Usage with PEFT ===

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import get_peft_model, OFTConfig

# Load a model quantized with Intel Neural Compressor FP8
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config={"method": "inc", "precision": "fp8"}
)

# Configure OFT
oft_config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    module_dropout=0.1,
    coft=False,
)

# Apply OFT - automatically uses IncOFTLinear for INC layers
peft_model = get_peft_model(model, oft_config)

# Fine-tune the model on Intel Habana Gaudi
# ... training code ...
</syntaxhighlight>

=== Usage on Intel Habana Gaudi ===

<syntaxhighlight lang="python">
import habana_frameworks.torch.core as htcore
from transformers import AutoModelForCausalLM
from peft import get_peft_model, OFTConfig

# Load model with INC FP8 quantization for Gaudi
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    device_map="hpu",  # Habana Processing Unit
)

# Apply FP8 quantization with INC
# (typically done during model preparation)

# Configure OFT
oft_config = OFTConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    module_dropout=0.05,
)

# Apply OFT
peft_model = get_peft_model(model, oft_config)

# Training on Gaudi
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

# Apply to INC FP8 quantized model
peft_model = get_peft_model(inc_quantized_model, oft_config)
</syntaxhighlight>

=== Handling Merge Operations ===

<syntaxhighlight lang="python">
# Note: Merging is not yet supported for INC layers
try:
    peft_model.merge_adapter()
except NotImplementedError as e:
    print(e)  # "Merging OFT with INC layers is not yet implemented"

# Similarly for unmerge
try:
    peft_model.unmerge_adapter()
except NotImplementedError as e:
    print(e)  # "Unmerging OFT from INC layers is not yet implemented"

# For inference, keep adapters as separate modules
peft_model.eval()
with torch.no_grad():
    output = peft_model(input_ids)
</syntaxhighlight>

=== Explicit Dispatcher Usage ===

<syntaxhighlight lang="python">
from peft.tuners.oft.inc import dispatch_inc
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import PatchedLinear

# Assume we have an INC FP8 quantized layer
inc_layer = PatchedLinear(...)

# Dispatch to create OFT adapter
oft_layer = dispatch_inc(
    target=inc_layer,
    adapter_name="default",
    r=8,
    oft_block_size=0,
    module_dropout=0.1
)

# Use the OFT-wrapped layer
output = oft_layer(input_tensor)
</syntaxhighlight>

== Implementation Details ==

=== Inheritance Structure ===

<code>IncOFTLinear</code> inherits from the standard <code>Linear</code> OFT layer implementation:

<syntaxhighlight lang="python">
class IncOFTLinear(Linear):
    def __init__(self, base_layer: torch.nn.Module, adapter_name: str, **kwargs):
        super().__init__(base_layer, adapter_name, **kwargs)
</syntaxhighlight>

This means it has all the functionality of the standard OFT Linear layer, with only merge/unmerge explicitly disabled.

=== Forward Pass ===

The forward pass is inherited from the parent <code>Linear</code> class and implements:

1. Check if adapters are disabled or merged
2. For each active adapter, apply OFT rotation transformation
3. Pass transformed input through INC FP8 quantized base layer
4. Handle dtype conversions as needed

=== PatchedLinear Detection ===

The dispatcher checks for INC's <code>PatchedLinear</code> module:

<syntaxhighlight lang="python">
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import PatchedLinear

if isinstance(target_base_layer, PatchedLinear):
    new_module = IncOFTLinear(target, adapter_name, **kwargs)
</syntaxhighlight>

=== Test Location ===

Tests for INC integration are not in the main PEFT repository. According to the file header:

<syntaxhighlight lang="python">
# NOTE: PEFT tests related to INC are handled under Optimum-Habana repository:
# - LLMs: https://github.com/huggingface/optimum-habana/blob/main/tests/test_peft_inference.py
# - Diffusers: https://github.com/huggingface/optimum-habana/blob/main/tests/test_diffusers.py
</syntaxhighlight>

=== Merge/Unmerge Not Implemented ===

Both merge and unmerge operations raise <code>NotImplementedError</code>:

<syntaxhighlight lang="python">
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    raise NotImplementedError("Merging OFT with INC layers is not yet implemented")

def unmerge(self) -> None:
    raise NotImplementedError("Unmerging OFT from INC layers is not yet implemented")
</syntaxhighlight>

This is likely due to the complexity of merging with FP8 quantized weights or because the use case on Habana hardware doesn't require merging.

== Related Pages ==

* [[huggingface_peft_OFTConfig|OFTConfig]] - Configuration class for OFT
* [[huggingface_peft_OFTModel|OFTModel]] - Main OFT model implementation
* [[huggingface_peft_OFT_AQLM|OFT AQLM Integration]] - Similar integration for AQLM quantization
* [[huggingface_peft_OFT_AWQ|OFT AWQ Integration]] - Similar integration for AWQ quantization
* [[huggingface_peft_OFT_GPTQ|OFT GPTQ Integration]] - Similar integration for GPTQ quantization
* [[huggingface_peft_OFT_EETQ|OFT EETQ Integration]] - Similar integration for EETQ quantization
* [[huggingface_peft_OFT_HQQ|OFT HQQ Integration]] - Similar integration for HQQ quantization

== See Also ==

* OFT Paper: [https://arxiv.org/abs/2306.07280 Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* Intel Neural Compressor: [https://github.com/intel/neural-compressor Intel Neural Compressor on GitHub]
* Optimum Habana: [https://github.com/huggingface/optimum-habana Optimum Habana on GitHub]
* PEFT Documentation: [https://huggingface.co/docs/peft HuggingFace PEFT Docs]
* Intel Habana Gaudi: [https://habana.ai/ Habana AI]
