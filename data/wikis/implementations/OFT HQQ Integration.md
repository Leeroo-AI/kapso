= OFT HQQ Integration =

== Knowledge Sources ==

* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* [https://arxiv.org/abs/2306.07280 OFT: Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* [https://github.com/mobiusml/hqq Half-Quadratic Quantization (HQQ)]

== Domains ==

[[Category:NLP]]
[[Category:PEFT]]
[[Category:Orthogonal_Fine_Tuning]]
[[Category:Quantization]]
[[Category:HQQ]]

== Overview ==

=== Description ===

The <code>HqqOFTLinear</code> class provides an implementation of Orthogonal Fine-Tuning (OFT) for HQQ (Half-Quadratic Quantization) quantized linear layers. This integration enables parameter-efficient fine-tuning of models quantized with HQQ, offering unique support for merging and unmerging adapters with quantized weights.

HQQ is a quantization method that uses half-quadratic optimization for weight quantization. Unlike most other quantization backends, the HQQ integration supports merging OFT adapter weights into the quantized base weights through a dequantize-transform-requantize process.

Key features:
* Supports HQQ quantized models with OFT adapters
* Unique merge/unmerge support for quantized weights
* Dequantizes, applies transformation, and requantizes for merging
* Preserves quantization configuration during merge operations
* Handles dtype conversions appropriately for autocast scenarios

=== Usage ===

This module is used internally by the PEFT library when applying OFT to HQQ-quantized models. The dispatcher function <code>dispatch_hqq</code> automatically detects HQQ quantized layers and wraps them with the appropriate OFT adapter.

== Code Reference ==

=== Source Location ===

File: <code>src/peft/tuners/oft/hqq.py</code>

Repository: HuggingFace PEFT (Parameter-Efficient Fine-Tuning)

=== Class: HqqOFTLinear ===

==== Signature ====

<syntaxhighlight lang="python">
class HqqOFTLinear(torch.nn.Module, OFTLayer):
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        r: int = 8,
        oft_block_size: int = 0,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        **kwargs,
    )
</syntaxhighlight>

==== Import ====

<syntaxhighlight lang="python">
from peft.tuners.oft.hqq import HqqOFTLinear, dispatch_hqq
</syntaxhighlight>

== I/O Contract ==

=== Constructor Parameters ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| base_layer || torch.nn.Module || required || The HQQ quantized linear layer to wrap
|-
| adapter_name || str || required || Name identifier for this adapter
|-
| r || int || 8 || OFT rank (number of OFT blocks per injected layer)
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
|}

=== Forward Method ===

{| class="wikitable"
! Input !! Type !! Description
|-
| x || torch.Tensor || Input tensor to the layer
|-
| *args || Any || Additional positional arguments
|-
| **kwargs || Any || Additional keyword arguments (including optional adapter_names)
|}

{| class="wikitable"
! Output !! Type !! Description
|-
| result || torch.Tensor || Output tensor after applying OFT transformation and quantized computation
|}

=== Merge Method ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| safe_merge || bool || False || If True, check for NaNs before merging
|-
| adapter_names || Optional[list[str]] || None || List of adapter names to merge (None = all active)
|}

=== Unmerge Method ===

No parameters - unmerges all merged adapters.

=== dispatch_hqq Function ===

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
| new_module || Optional[torch.nn.Module] || HqqOFTLinear instance if target is HQQ quantized, None otherwise
|}

== Usage Examples ==

=== Basic Usage with PEFT ===

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import get_peft_model, OFTConfig

# Load a model quantized with HQQ
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config={"method": "hqq"}
)

# Configure OFT
oft_config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    module_dropout=0.1,
    coft=False,
)

# Apply OFT - automatically uses HqqOFTLinear for HQQ layers
peft_model = get_peft_model(model, oft_config)

# Fine-tune the model
# ... training code ...
</syntaxhighlight>

=== Merging Adapters (Unique to HQQ) ===

<syntaxhighlight lang="python">
from peft import get_peft_model, OFTConfig

# Load HQQ model and add OFT adapters
peft_model = get_peft_model(hqq_model, oft_config)

# Train the model
# ... training code ...

# Merge adapters into base weights (supported for HQQ!)
peft_model.merge_adapter()

# Model now has adapters merged into quantized weights
# Can be used for inference without adapter overhead

# Later, can unmerge if needed
peft_model.unmerge_adapter()
</syntaxhighlight>

=== Safe Merging with NaN Check ===

<syntaxhighlight lang="python">
from peft import get_peft_model, OFTConfig

peft_model = get_peft_model(hqq_model, oft_config)

# Train the model
# ... training code ...

# Safely merge with NaN checking
try:
    peft_model.merge_adapter(safe_merge=True)
    print("Merge successful!")
except ValueError as e:
    print(f"Merge failed: {e}")
    # Handle broken adapter
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

# Apply to HQQ quantized model
peft_model = get_peft_model(hqq_quantized_model, oft_config)
</syntaxhighlight>

=== Selective Adapter Merging ===

<syntaxhighlight lang="python">
from peft import PeftModel

# Model with multiple adapters
peft_model = PeftModel(hqq_model, oft_config)
peft_model.load_adapter("adapter1", adapter_name="task1")
peft_model.load_adapter("adapter2", adapter_name="task2")

# Merge only specific adapters
peft_model.merge_adapter(adapter_names=["task1"])

# task1 is now merged, task2 still separate
</syntaxhighlight>

== Implementation Details ==

=== Merge Operation ===

The merge process is unique to HQQ and involves:

1. Dequantize the base weights:
<syntaxhighlight lang="python">
output = layer.dequantize()
</syntaxhighlight>

2. Get the OFT transformation matrix:
<syntaxhighlight lang="python">
oft_data = self.get_delta_weight(active_adapter)
</syntaxhighlight>

3. Apply the transformation:
<syntaxhighlight lang="python">
output = torch.transpose(output, 0, 1)
w_data = torch.mm(oft_data, output.to(oft_data.dtype))
w_data = torch.transpose(w_data, 0, 1)
</syntaxhighlight>

4. Requantize with original configuration:
<syntaxhighlight lang="python">
new_hqq_layer = HQQLinear(None, quant_config, compute_dtype=layer.compute_dtype, device=layer.device)
new_hqq_layer.quantize(w_data, **quant_config)
self.base_layer = new_hqq_layer
</syntaxhighlight>

=== Unmerge Operation ===

The unmerge process reverses the transformation:

1. Dequantize current (merged) weights
2. Apply inverse OFT transformation using <code>oft_data.t()</code> (transpose)
3. Requantize back to HQQ format

=== Forward Pass Logic ===

The forward method implements:

1. Check if adapters are disabled:
   * If merged, unmerge first
   * Return base layer result directly
2. If adapters are merged, use base layer directly
3. Otherwise, for each active adapter:
   * Apply dtype conversion if not in autocast mode
   * Apply the OFT rotation transformation (oft_R)
4. Pass transformed input through HQQ quantized base layer
5. Convert result back to expected dtype if needed

=== Quantization Configuration Preservation ===

The implementation carefully preserves the quantization configuration during merge/unmerge:

<syntaxhighlight lang="python">
quant_config = {
    **copy.deepcopy(layer.quant_config),
    "offload_meta": layer.offload_meta
}
# ... transform ...
quant_config.pop("offload_meta", None)
new_hqq_layer.quantize(w_data, **quant_config)
</syntaxhighlight>

=== Fan-in/Fan-out ===

The class explicitly sets <code>fan_in_fan_out = False</code>, meaning it expects standard weight layout.

== Related Pages ==

* [[huggingface_peft_OFTConfig|OFTConfig]] - Configuration class for OFT
* [[huggingface_peft_OFTModel|OFTModel]] - Main OFT model implementation
* [[huggingface_peft_OFT_AQLM|OFT AQLM Integration]] - Similar integration for AQLM quantization
* [[huggingface_peft_OFT_AWQ|OFT AWQ Integration]] - Similar integration for AWQ quantization
* [[huggingface_peft_OFT_GPTQ|OFT GPTQ Integration]] - Similar integration for GPTQ quantization
* [[huggingface_peft_OFT_EETQ|OFT EETQ Integration]] - Similar integration for EETQ quantization

== See Also ==

* OFT Paper: [https://arxiv.org/abs/2306.07280 Controlling Text-to-Image Diffusion by Orthogonal Finetuning]
* HQQ Repository: [https://github.com/mobiusml/hqq HQQ on GitHub]
* PEFT Documentation: [https://huggingface.co/docs/peft HuggingFace PEFT Docs]
