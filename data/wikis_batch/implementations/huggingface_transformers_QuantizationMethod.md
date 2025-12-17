{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Enumeration of all supported quantization methods in HuggingFace Transformers for model compression and optimization.

=== Description ===

The QuantizationMethod enum defines string identifiers for all quantization backends supported by Transformers. These identifiers are used throughout the quantization pipeline to dispatch to the correct configuration classes and quantizer implementations.

=== Usage ===

This enum is used internally when parsing quantization configurations, selecting quantizers, and validating model compatibility. Users typically don't instantiate this directly but encounter these string values in model configs and quantization_config parameters.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/utils/quantization_config.py

=== Signature ===
<syntaxhighlight lang="python">
class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    VPTQ = "vptq"
    QUANTO = "quanto"
    EETQ = "eetq"
    HIGGS = "higgs"
    HQQ = "hqq"
    COMPRESSED_TENSORS = "compressed-tensors"
    FBGEMM_FP8 = "fbgemm_fp8"
    TORCHAO = "torchao"
    BITNET = "bitnet"
    SPQR = "spqr"
    FP8 = "fp8"
    QUARK = "quark"
    FPQUANT = "fp_quant"
    AUTOROUND = "auto-round"
    MXFP4 = "mxfp4"
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.utils.quantization_config import QuantizationMethod
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| N/A || N/A || N/A || Enum class, no inputs required
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| method_value || str || String identifier for the quantization method
|}

== Usage Examples ==

<syntaxhighlight lang="python">
from transformers.utils.quantization_config import QuantizationMethod

# Access method identifiers
method = QuantizationMethod.BITS_AND_BYTES
print(method)  # "bitsandbytes"

# Check available methods
all_methods = [m.value for m in QuantizationMethod]
print(all_methods)
# ['bitsandbytes', 'gptq', 'awq', 'aqlm', 'vptq', 'quanto', ...]

# Use in configuration
from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(load_in_4bit=True)
assert config.quant_method == QuantizationMethod.BITS_AND_BYTES
</syntaxhighlight>

== Method Descriptions ==

{| class="wikitable"
|-
! Method !! Description !! Typical Use Case
|-
| BITS_AND_BYTES || LLM.int8() and 4-bit NF4/FP4 quantization || Easy GPU inference, QLoRA fine-tuning
|-
| GPTQ || Post-training quantization using Hessian information || High accuracy 2-4 bit inference
|-
| AWQ || Activation-aware weight quantization || Optimized 4-bit inference
|-
| AQLM || Additive quantization with codebooks || Research, extreme compression
|-
| QUANTO || PyTorch native quantization || Cross-platform, flexible
|-
| EETQ || Efficient INT8 quantization || Fast INT8 inference
|-
| HQQ || Half-quadratic quantization || Low-bit quantization
|-
| COMPRESSED_TENSORS || Flexible scheme-based quantization || Advanced compression pipelines
|-
| FP8 || 8-bit floating point || H100/MI300 GPU acceleration
|-
| TORCHAO || PyTorch AO quantization API || Native PyTorch workflows
|-
| BITNET || 1-bit extreme quantization || Extreme memory reduction
|-
| SPQR || Sparse quantization || Outlier-heavy models
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Quantization_Method_Selection]]

=== Used By ===
* [[used_by::Implementation:huggingface_transformers_BitsAndBytesConfig]]
* [[used_by::Implementation:huggingface_transformers_get_hf_quantizer_init]]
