# Implementation: DeepSeek Registry

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Model_Support]], [[domain::Registry]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

The `_deepseek.py` module implements the DeepSeek model family registration for the Unsloth model registry. It defines metadata classes and registration functions for various DeepSeek models including DeepSeek-V3, DeepSeek-R1, and their distilled variants (Llama and Qwen-based).

Key responsibilities:
* Define custom ModelInfo subclasses for DeepSeek naming conventions
* Configure ModelMeta for each DeepSeek variant with supported quantization types
* Provide idempotent registration functions for each model family
* Auto-register all DeepSeek models on module import

== Code Reference ==

'''File:''' `unsloth/registry/_deepseek.py` (206 lines)

=== Custom ModelInfo Classes ===

<syntaxhighlight lang="python">
class DeepseekV3ModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-V{version}"
        return super().construct_model_name(
            base_name, version, size, quant_type, instruct_tag, key
        )


class DeepseekR1ModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}" if version else base_name
        if size:
            key = f"{key}-{size}B"
        return super().construct_model_name(
            base_name, version, size, quant_type, instruct_tag, key
        )
</syntaxhighlight>

=== Model Metadata Definitions ===

<syntaxhighlight lang="python">
DeepseekV3Meta = ModelMeta(
    org = "deepseek-ai",
    base_name = "DeepSeek",
    instruct_tags = [None],
    model_version = "3",
    model_sizes = [""],
    model_info_cls = DeepseekV3ModelInfo,
    is_multimodal = False,
    quant_types = [QuantType.NONE, QuantType.BF16],
)

DeepseekR1DistillQwenMeta = ModelMeta(
    org = "deepseek-ai",
    base_name = "DeepSeek-R1-Distill",
    instruct_tags = [None],
    model_version = "Qwen",
    model_sizes = ["1.5", "7", "14", "32"],
    model_info_cls = DeepseekR1ModelInfo,
    is_multimodal = False,
    quant_types = {
        "1.5": [QuantType.UNSLOTH, QuantType.BNB, QuantType.GGUF],
        "7": [QuantType.UNSLOTH, QuantType.BNB],
        "14": [QuantType.UNSLOTH, QuantType.BNB, QuantType.GGUF],
        "32": [QuantType.GGUF, QuantType.BNB],
    },
)
</syntaxhighlight>

== I/O Contract ==

=== Supported Model Families ===

{| class="wikitable"
|-
! Model Meta !! Organization !! Sizes !! Quantization Types
|-
| `DeepseekV3Meta` || deepseek-ai || N/A || NONE, BF16
|-
| `DeepseekV3_0324Meta` || deepseek-ai || N/A || NONE, GGUF
|-
| `DeepseekR1Meta` || deepseek-ai || N/A || NONE, BF16, GGUF
|-
| `DeepseekR1ZeroMeta` || deepseek-ai || N/A || NONE, GGUF
|-
| `DeepseekR1DistillLlamaMeta` || deepseek-ai || 8B, 70B || UNSLOTH, GGUF (8B); GGUF (70B)
|-
| `DeepseekR1DistillQwenMeta` || deepseek-ai || 1.5B, 7B, 14B, 32B || Varies by size
|}

=== Registration Functions ===

{| class="wikitable"
|-
! Function !! Description !! Idempotent Flag
|-
| `register_deepseek_v3_models()` || Register DeepSeek-V3 models || `_IS_DEEPSEEK_V3_REGISTERED`
|-
| `register_deepseek_v3_0324_models()` || Register DeepSeek-V3-0324 models || `_IS_DEEPSEEK_V3_0324_REGISTERED`
|-
| `register_deepseek_r1_models()` || Register DeepSeek-R1 models || `_IS_DEEPSEEK_R1_REGISTERED`
|-
| `register_deepseek_r1_zero_models()` || Register DeepSeek-R1-Zero models || `_IS_DEEPSEEK_R1_ZERO_REGISTERED`
|-
| `register_deepseek_r1_distill_llama_models()` || Register R1 Llama distillations || `_IS_DEEPSEEK_R1_DISTILL_LLAMA_REGISTERED`
|-
| `register_deepseek_r1_distill_qwen_models()` || Register R1 Qwen distillations || `_IS_DEEPSEEK_R1_DISTILL_QWEN_REGISTERED`
|-
| `register_deepseek_models()` || Register all DeepSeek models || N/A (calls all above)
|}

=== Function Signature ===

<syntaxhighlight lang="python">
def register_deepseek_models(include_original_model: bool = False) -> None:
    """
    Register all DeepSeek model families.

    Args:
        include_original_model: If True, also register original deepseek-ai models
                               in addition to unsloth quantized versions.
    """
</syntaxhighlight>

== Usage Examples ==

=== Register All DeepSeek Models ===

<syntaxhighlight lang="python">
from unsloth.registry._deepseek import register_deepseek_models

# Register only unsloth quantized versions
register_deepseek_models()

# Register both unsloth and original deepseek-ai versions
register_deepseek_models(include_original_model=True)
</syntaxhighlight>

=== Register Specific Model Family ===

<syntaxhighlight lang="python">
from unsloth.registry._deepseek import (
    register_deepseek_r1_distill_qwen_models,
    register_deepseek_r1_distill_llama_models,
)

# Register only distilled models
register_deepseek_r1_distill_qwen_models(include_original_model=True)
register_deepseek_r1_distill_llama_models(include_original_model=True)
</syntaxhighlight>

=== Check Registered Models ===

<syntaxhighlight lang="python">
from unsloth.registry.registry import MODEL_REGISTRY
from unsloth.registry._deepseek import register_deepseek_models

register_deepseek_models(include_original_model=True)

# List all registered DeepSeek models
for model_id in MODEL_REGISTRY:
    if "DeepSeek" in model_id or "deepseek" in model_id:
        print(model_id)

# Example output:
# unsloth/DeepSeek-V3
# unsloth/DeepSeek-V3-bf16
# unsloth/DeepSeek-R1
# unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit
# deepseek-ai/DeepSeek-V3
</syntaxhighlight>

=== Access Model Metadata ===

<syntaxhighlight lang="python">
from unsloth.registry.registry import MODEL_REGISTRY

model_info = MODEL_REGISTRY["unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"]
print(f"Organization: {model_info.org}")
print(f"Base name: {model_info.base_name}")
print(f"Size: {model_info.size}B")
print(f"Quant type: {model_info.quant_type}")
print(f"Full path: {model_info.model_path}")
</syntaxhighlight>

== Related Pages ==

* [[Unslothai_Unsloth_Model_Registry|Model Registry]] - Core registry API used by this module
* [[Unslothai_Unsloth_Device_Type|Device Type]] - Hardware detection for quantization support
