{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Quantization Documentation|https://huggingface.co/docs/transformers/main_classes/quantization]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete method for computing module exclusion list combining architecture defaults with user specifications provided by HuggingFace Transformers.

=== Description ===
HfQuantizer.get_modules_to_not_convert implements the Module_Targeting principle by merging multiple sources of module exclusion information: architecture-specific defaults (detected via get_keys_to_not_convert), user-provided skip lists, and optionally additional full-precision modules. The method handles list deduplication and provides control over whether to include default skips, allowing quantization backends to customize skip behavior while maintaining consistent semantics.

=== Usage ===
Call this method during preprocessing or conversion to obtain the final list of modules that should remain in full precision. Pass the result to module replacement or quantization loops to check each module against the skip list. Override or extend in custom quantizers to implement backend-specific skip logic. Use add_default_skips=True to include architecture defaults alongside user specifications.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/quantizers/base.py
* '''Lines:''' 256-276

=== Signature ===
<syntaxhighlight lang="python">
class HfQuantizer(ABC):
    """
    Abstract class of the HuggingFace quantizer.
    """

    @staticmethod
    def get_modules_to_not_convert(
        model: "PreTrainedModel",
        skip_modules: list[str] | None = None,
        keep_in_fp32_modules: list[str] | None = None,
        add_default_skips: bool = False,
    ):
        """
        Compute the list of modules to exclude from quantization.

        Args:
            model (~transformers.PreTrainedModel):
                The model to analyze for skip modules.
            skip_modules (list[str], optional):
                User-provided list of module names to skip. If None and add_default_skips=False,
                only architecture defaults are used.
            keep_in_fp32_modules (list[str], optional):
                Additional modules to keep in FP32. Merged with skip_modules.
            add_default_skips (bool, optional):
                If True, include architecture-detected defaults even when skip_modules is provided.
                If False and skip_modules is provided, only use skip_modules.
                Defaults to False.

        Returns:
            list[str]: Deduplicated list of module names to exclude from quantization.
        """
        if skip_modules is None or add_default_skips:
            modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            modules_to_not_convert = []

        if skip_modules is not None:
            modules_to_not_convert.extend(skip_modules)

        if keep_in_fp32_modules is not None:
            modules_to_not_convert.extend(keep_in_fp32_modules)

        modules_to_not_convert = list(set(modules_to_not_convert))

        return modules_to_not_convert


def get_keys_to_not_convert(model) -> list:
    """
    Automatically detect critical modules that should not be quantized.

    Args:
        model (~transformers.PreTrainedModel): The model to analyze.

    Returns:
        list[str]: Module names that should remain in full precision.
    """
    # Remove tied weights
    tied_keys = set()
    if len(model.all_tied_weights_keys) > 0:
        tied_keys = set(model.all_tied_weights_keys.values()) | set(model.all_tied_weights_keys.keys())

    # Remove last module
    last_module_key = {list(model.named_parameters())[-1][0]}

    # Remove output embedding module
    output_emb_module = model.get_output_embeddings()
    output_emb_keys = {
        name
        for name, module in model.named_modules()
        if output_emb_module is not None and id(module) == id(output_emb_module)
    }
    modules_to_not_convert = tied_keys | last_module_key | output_emb_keys

    modules_to_not_convert = list({k.removesuffix(".weight") for k in modules_to_not_convert})

    return list(modules_to_not_convert)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer, get_keys_to_not_convert
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Model to analyze for critical modules
|-
| skip_modules || list[str] || No || User-specified module names to skip
|-
| keep_in_fp32_modules || list[str] || No || Additional modules for full precision
|-
| add_default_skips || bool || No || Include architecture defaults (default: False)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| modules_to_not_convert || list[str] || Deduplicated list of module names to exclude from quantization
|}

== Usage Examples ==

=== Architecture Defaults Only ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from transformers.quantizers.base import HfQuantizer, get_keys_to_not_convert

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Get architecture-specific defaults
defaults = get_keys_to_not_convert(model)
print("Default skip modules:", defaults)
# Output: ['lm_head', 'transformer.wte']
# - lm_head: output head (last module + output embeddings)
# - transformer.wte: input embeddings (tied with lm_head)

# Use in quantization
skip_list = HfQuantizer.get_modules_to_not_convert(
    model,
    skip_modules=None,  # No user overrides
    add_default_skips=False,  # Not needed (skip_modules is None)
)
print("Final skip list:", skip_list)
# Output: ['lm_head', 'transformer.wte']
</syntaxhighlight>

=== User-Specified Modules Only ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from transformers.quantizers.base import HfQuantizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# User wants to skip specific modules, ignoring defaults
user_skips = [
    "model.layers.31.mlp.gate_proj",  # Last layer gate
    "model.layers.31.mlp.up_proj",    # Last layer up
]

skip_list = HfQuantizer.get_modules_to_not_convert(
    model,
    skip_modules=user_skips,
    add_default_skips=False,  # Don't include architecture defaults
)

print("Skip list:", skip_list)
# Output: ['model.layers.31.mlp.gate_proj', 'model.layers.31.mlp.up_proj']
# Note: lm_head NOT included (defaults ignored)
</syntaxhighlight>

=== Merging User and Defaults ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from transformers.quantizers.base import HfQuantizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# User wants additional skips beyond defaults
user_skips = [
    "model.norm",  # Final layer norm
    "model.layers.0.self_attn.q_proj",  # First layer attention (for debugging)
]

skip_list = HfQuantizer.get_modules_to_not_convert(
    model,
    skip_modules=user_skips,
    add_default_skips=True,  # Include architecture defaults
)

print("Skip list:", skip_list)
# Output includes both defaults and user skips:
# ['lm_head', 'model.embed_tokens', 'model.norm', 'model.layers.0.self_attn.q_proj']
</syntaxhighlight>

=== Using keep_in_fp32_modules ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.quantizers.auto import AutoHfQuantizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure quantization with skip modules
config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_skip_modules=["lm_head"],  # Stored in config
)

# Additional FP32 modules (e.g., for fine-tuning adapters)
fp32_modules = ["model.layers.0.adapter", "model.layers.1.adapter"]

quantizer = AutoHfQuantizer.from_config(config)
skip_list = quantizer.get_modules_to_not_convert(
    model,
    skip_modules=config.llm_int8_skip_modules,
    keep_in_fp32_modules=fp32_modules,
    add_default_skips=True,
)

print("Final skip list:", skip_list)
# Includes: architecture defaults + config skips + fp32 modules
</syntaxhighlight>

=== Integration in Quantization Loop ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.quantizers.base import HfQuantizer
import torch.nn as nn

model = AutoModelForCausalLM.from_pretrained("gpt2")
config = BitsAndBytesConfig(load_in_4bit=True)

# Get skip list
skip_modules = HfQuantizer.get_modules_to_not_convert(
    model,
    skip_modules=None,
    add_default_skips=False,
)

print(f"Modules to skip: {skip_modules}")

# Quantization loop
quantized_count = 0
skipped_count = 0

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        if name in skip_modules:
            print(f"SKIP: {name} (in skip list)")
            skipped_count += 1
        else:
            print(f"QUANTIZE: {name}")
            # quantize_module(module, config)
            quantized_count += 1

print(f"\nSummary: {quantized_count} quantized, {skipped_count} skipped")
</syntaxhighlight>

=== Detecting Tied Weights ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from transformers.quantizers.base import get_keys_to_not_convert

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Check tied weights
print("Tied weights:")
if hasattr(model, "all_tied_weights_keys"):
    for key, value in model.all_tied_weights_keys.items():
        print(f"  {key} ↔ {value}")

# Output:
# transformer.wte.weight ↔ lm_head.weight

# get_keys_to_not_convert automatically detects these
defaults = get_keys_to_not_convert(model)
print("\nAuto-detected skip modules:", defaults)
# ['lm_head', 'transformer.wte']
# Both tied modules are excluded to prevent issues
</syntaxhighlight>

=== Custom Skip Logic ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer
import torch.nn as nn

class CustomQuantizer(HfQuantizer):
    def get_custom_skip_list(self, model):
        """Custom logic for determining skip modules"""
        skip_list = self.get_modules_to_not_convert(
            model,
            skip_modules=self.quantization_config.skip_modules,
            add_default_skips=True,
        )

        # Add all LayerNorm modules
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                skip_list.append(name)
                print(f"Auto-skip LayerNorm: {name}")

        # Add all Embedding modules
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                skip_list.append(name)
                print(f"Auto-skip Embedding: {name}")

        # Deduplicate
        return list(set(skip_list))

# Usage
quantizer = CustomQuantizer(config)
skip_list = quantizer.get_custom_skip_list(model)
# Includes: defaults + user skips + all LayerNorm + all Embedding
</syntaxhighlight>

=== Analyzing Memory Impact of Skip Modules ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from transformers.quantizers.base import HfQuantizer
import torch

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Get skip list
skip_modules = HfQuantizer.get_modules_to_not_convert(
    model,
    skip_modules=None,
    add_default_skips=False,
)

# Calculate parameter counts
total_params = 0
skip_params = 0
quantize_params = 0

for name, param in model.named_parameters():
    module_name = name.removesuffix(".weight").removesuffix(".bias")
    param_count = param.numel()
    total_params += param_count

    if module_name in skip_modules:
        skip_params += param_count
        print(f"SKIP: {name} ({param_count:,} params)")
    else:
        quantize_params += param_count

print(f"\nTotal parameters: {total_params:,}")
print(f"Skip parameters: {skip_params:,} ({100*skip_params/total_params:.2f}%)")
print(f"Quantize parameters: {quantize_params:,} ({100*quantize_params/total_params:.2f}%)")

# Estimate memory savings
fp16_memory = total_params * 2  # 2 bytes per param
int4_memory = (quantize_params * 0.5) + (skip_params * 2)  # 0.5 bytes for quantized, 2 for skipped

print(f"\nMemory (FP16): {fp16_memory / 1e9:.2f} GB")
print(f"Memory (INT4 with skips): {int4_memory / 1e9:.2f} GB")
print(f"Savings: {100 * (1 - int4_memory/fp16_memory):.1f}%")
</syntaxhighlight>

=== BitsAndBytes Config Integration ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.quantizers.auto import AutoHfQuantizer

# Configure skip modules in BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=["lm_head", "model.norm"],  # User-specified skips
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=config,
    device_map="auto",
)

# Inside from_pretrained, skip modules are used:
quantizer = AutoHfQuantizer.from_config(config)
skip_list = quantizer.get_modules_to_not_convert(
    model,
    skip_modules=config.llm_int8_skip_modules,
    add_default_skips=True,  # Merge with architecture defaults
)

# Model loads with specified modules in full precision
# All other Linear layers are quantized to INT8
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Module_Targeting]]

=== Used By ===
* [[used_by::Implementation:huggingface_transformers_Quantizer_convert_weights]]
* [[used_by::Implementation:huggingface_transformers_Quantizer_preprocess]]

=== Related ===
* [[related::Implementation:huggingface_transformers_BitsAndBytesConfig_setup]]
