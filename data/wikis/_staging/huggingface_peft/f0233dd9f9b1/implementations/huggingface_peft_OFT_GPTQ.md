= OFT GPTQ Integration =

== Knowledge Sources ==

* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* [https://arxiv.org/abs/2306.07280 OFT: Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* [https://arxiv.org/abs/2210.17323 GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers]

== Domains ==

[[Category:NLP]]
[[Category:PEFT]]
[[Category:Orthogonal_Fine_Tuning]]
[[Category:Quantization]]
[[Category:GPTQ]]

== Overview ==

=== Description ===

The <code>GPTQOFTLinear</code> class provides an implementation of Orthogonal Fine-Tuning (OFT) for GPTQ (Generative Pre-trained Transformer Quantization) quantized linear layers. This integration enables parameter-efficient fine-tuning of models quantized with GPTQ, allowing practitioners to fine-tune 4-bit quantized models while maintaining memory efficiency.

GPTQ is a post-training quantization method specifically designed for generative pre-trained transformers, achieving high compression ratios with minimal accuracy loss. The OFT adapter wraps GPTQ quantized layers, allowing fine-tuning while preserving the quantized structure.

Key features:
* Supports both GPTQModel and AutoGPTQ quantized models
* Preserves quantization during fine-tuning
* Applies orthogonal transformations before quantized computation
* Handles dtype conversions appropriately for autocast scenarios
* Note: Merging adapters into GPTQ weights is not supported

=== Usage ===

This module is used internally by the PEFT library when applying OFT to GPTQ-quantized models. The dispatcher function <code>dispatch_gptq</code> automatically detects GPTQ quantized layers and wraps them with the appropriate OFT adapter.

== Code Reference ==

=== Source Location ===

File: <code>src/peft/tuners/oft/gptq.py</code>

Repository: HuggingFace PEFT (Parameter-Efficient Fine-Tuning)

=== Class: GPTQOFTLinear ===

==== Signature ====

<syntaxhighlight lang="python">
class GPTQOFTLinear(torch.nn.Module, OFTLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 8,
        oft_block_size: int = 0,
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        fan_in_fan_out: bool = False,
        init_weights: bool = True,
        **kwargs,
    )
</syntaxhighlight>

==== Import ====

<syntaxhighlight lang="python">
from peft.tuners.oft.gptq import GPTQOFTLinear, dispatch_gptq
</syntaxhighlight>

== I/O Contract ==

=== Constructor Parameters ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| base_layer || Module || required || The GPTQ quantized linear layer to wrap
|-
| adapter_name || str || required || Name identifier for this adapter
|-
| r || int || 8 || OFT rank (number of OFT blocks per injected layer)
|-
| oft_block_size || int || 0 || Size of OFT blocks across different layers
|-
| module_dropout || float || 0.0 || Multiplicative dropout probability for OFT blocks
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
|-
| init_weights || bool || True || Whether to initialize OFT weights
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

=== dispatch_gptq Function ===

{| class="wikitable"
! Parameter !! Type !! Description
|-
| target || torch.nn.Module || The target module to potentially wrap
|-
| adapter_name || str || Name for the adapter
|-
| **kwargs || Any || Additional keyword arguments (may include gptq_quantization_config)
|}

{| class="wikitable"
! Return !! Type !! Description
|-
| new_module || Optional[torch.nn.Module] || GPTQOFTLinear instance if target is GPTQ quantized, None otherwise
|}

== Usage Examples ==

=== Basic Usage with PEFT ===

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import get_peft_model, OFTConfig

# Load a model quantized with GPTQ
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config={"bits": 4, "method": "gptq"}
)

# Configure OFT
oft_config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    module_dropout=0.1,
    coft=False,
)

# Apply OFT - automatically uses GPTQOFTLinear for GPTQ layers
peft_model = get_peft_model(model, oft_config)

# Fine-tune the model
# ... training code ...
</syntaxhighlight>

=== Loading Pre-quantized GPTQ Model ===

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import get_peft_model, OFTConfig

# Load pre-quantized GPTQ model from HuggingFace Hub
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto",
    trust_remote_code=False,
)

# Apply OFT for fine-tuning
oft_config = OFTConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    module_dropout=0.05,
)

peft_model = get_peft_model(model, oft_config)
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

# Apply to GPTQ quantized model
peft_model = get_peft_model(gptq_quantized_model, oft_config)
</syntaxhighlight>

=== Explicit Dispatcher Usage ===

<syntaxhighlight lang="python">
from peft.tuners.oft.gptq import dispatch_gptq

# Assume we have a GPTQ quantized layer
# Works with both gptqmodel.nn_modules.qlinear.BaseQuantLinear
# and auto_gptq quantized linear layers

# Dispatch to create OFT adapter
oft_layer = dispatch_gptq(
    target=gptq_layer,
    adapter_name="default",
    r=8,
    oft_block_size=0,
    module_dropout=0.1,
    gptq_quantization_config=quantization_config,
)

# Use the OFT-wrapped layer
output = oft_layer(input_tensor)
</syntaxhighlight>

=== Important: Merging Not Supported ===

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import get_peft_model, OFTConfig

# Load and configure GPTQ model with OFT
model = AutoModelForCausalLM.from_pretrained("gptq-model")
oft_config = OFTConfig(r=8, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(model, oft_config)

# Train the model
# ...

# IMPORTANT: Cannot merge adapters into GPTQ weights
try:
    peft_model.merge_adapter()
except ValueError as e:
    print(e)  # "Cannot merge OFT layers when the model is gptq quantized"
</syntaxhighlight>

== Implementation Details ==

=== Forward Pass Logic ===

The forward method implements the following logic:

1. First computes the result from the quantized base layer (note: this appears twice in the code, likely a bug)
2. Check if adapters are disabled - if so, return base layer result directly
3. For each active adapter:
   * Apply dtype conversion if not in autocast mode
   * Apply the OFT rotation transformation (oft_R)
4. Pass transformed input through the GPTQ quantized base layer
5. Convert result back to expected dtype if needed

=== Multiple Backend Support ===

The dispatcher supports two GPTQ implementations:

1. '''GPTQModel''' (preferred):
<syntaxhighlight lang="python">
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
</syntaxhighlight>

2. '''AutoGPTQ''' (fallback):
<syntaxhighlight lang="python">
from peft.utils import get_auto_gptq_quant_linear
quant_linear = get_auto_gptq_quant_linear(cfg)
</syntaxhighlight>

=== Quantization Weight Handling ===

The dispatcher updates the <code>qweight</code> attribute to reference the GPTQ quantized weights:

<syntaxhighlight lang="python">
target.qweight = target_base_layer.qweight
</syntaxhighlight>

This ensures proper access to the quantized weights used by GPTQ.

=== Backward Compatibility ===

The class maintains both <code>base_layer</code> and <code>quant_linear_module</code> attributes pointing to the same object:
* <code>base_layer</code> - for consistency with other OFT implementations
* <code>quant_linear_module</code> - for backward compatibility with older code

=== Merging Restrictions ===

The OFTModel class explicitly prevents merging for GPTQ models:

<syntaxhighlight lang="python">
if getattr(self.model, "quantization_method", None) == "gptq":
    raise ValueError("Cannot merge OFT layers when the model is gptq quantized")
</syntaxhighlight>

This is because merging would require dequantization and re-quantization, which is not supported.

== Related Pages ==

* [[huggingface_peft_OFTConfig|OFTConfig]] - Configuration class for OFT
* [[huggingface_peft_OFTModel|OFTModel]] - Main OFT model implementation
* [[huggingface_peft_OFT_AQLM|OFT AQLM Integration]] - Similar integration for AQLM quantization
* [[huggingface_peft_OFT_AWQ|OFT AWQ Integration]] - Similar integration for AWQ quantization
* [[huggingface_peft_OFT_EETQ|OFT EETQ Integration]] - Similar integration for EETQ quantization
* [[huggingface_peft_OFT_HQQ|OFT HQQ Integration]] - Similar integration for HQQ quantization

== See Also ==

* OFT Paper: [https://arxiv.org/abs/2306.07280 Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* GPTQ Paper: [https://arxiv.org/abs/2210.17323 GPTQ: Accurate Post-Training Quantization]
* PEFT Documentation: [https://huggingface.co/docs/peft HuggingFace PEFT Docs]
* GPTQModel: [https://github.com/ModelCloud/GPTQModel GPTQModel on GitHub]
