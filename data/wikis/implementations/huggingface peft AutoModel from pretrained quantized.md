{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Quantization|https://huggingface.co/docs/transformers/main_classes/quantization]]
* [[source::Repo|bitsandbytes|https://github.com/bitsandbytes-foundation/bitsandbytes]]
|-
! Domains
| [[domain::Model_Loading]], [[domain::Quantization]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for loading transformer models with bitsandbytes 4-bit/8-bit quantization applied during loading.

=== Description ===

`AutoModelForCausalLM.from_pretrained()` with a `quantization_config` parameter applies on-the-fly quantization. The model weights are loaded, quantized, and placed on GPU in a single operation. Linear layers are replaced with `Linear4bit` or `Linear8bitLt` variants.

=== Usage ===

Pass a BitsAndBytesConfig to `from_pretrained()` with `device_map="auto"` to load a quantized model ready for QLoRA training.

== Code Reference ==

=== Source Location ===
* '''Library:''' `transformers.AutoModelForCausalLM` (external)
* '''Integration:''' `bitsandbytes` for quantization

=== Signature ===
<syntaxhighlight lang="python">
@classmethod
def from_pretrained(
    cls,
    pretrained_model_name_or_path: str,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    device_map: Optional[Union[str, dict]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> PreTrainedModel:
    """
    Load model with optional quantization.

    Args:
        pretrained_model_name_or_path: Model ID or path
        quantization_config: BitsAndBytesConfig for 4-bit/8-bit
        device_map: Must be "auto" for quantization

    Returns:
        Quantized model with Linear4bit/Linear8bitLt layers
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
</syntaxhighlight>

== Usage Examples ==

=== 4-bit NF4 Loading ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",  # Required for quantization
)
</syntaxhighlight>

=== 8-bit Loading ===
<syntaxhighlight lang="python">
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Quantized_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Quantization_Environment]]
