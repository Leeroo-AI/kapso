# Implementation: Model Registry

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

The `registry.py` module provides the core model registry API for Unsloth. It defines data classes for model metadata, quantization types, and a global registry for tracking supported models. The registry enables consistent model naming, path construction, and validation across the library.

Key responsibilities:
* Define quantization type enumeration (BNB, UNSLOTH, GGUF, BF16, NONE)
* Provide `ModelInfo` and `ModelMeta` dataclasses for model metadata
* Maintain a global `MODEL_REGISTRY` dictionary of registered models
* Offer registration functions with duplicate detection and validation

== Code Reference ==

'''File:''' `unsloth/registry/registry.py` (191 lines)

=== Quantization Types ===

<syntaxhighlight lang="python">
class QuantType(Enum):
    BNB = "bnb"
    UNSLOTH = "unsloth"  # dynamic 4-bit quantization
    GGUF = "GGUF"
    NONE = "none"
    BF16 = "bf16"  # only for Deepseek V3

# Tags for Hugging Face model paths
BNB_QUANTIZED_TAG = "bnb-4bit"
UNSLOTH_DYNAMIC_QUANT_TAG = "unsloth" + "-" + BNB_QUANTIZED_TAG"
GGUF_TAG = "GGUF"
BF16_TAG = "bf16"

QUANT_TAG_MAP = {
    QuantType.BNB: BNB_QUANTIZED_TAG,
    QuantType.UNSLOTH: UNSLOTH_DYNAMIC_QUANT_TAG,
    QuantType.GGUF: GGUF_TAG,
    QuantType.NONE: None,
    QuantType.BF16: BF16_TAG,
}
</syntaxhighlight>

=== ModelInfo Dataclass ===

<syntaxhighlight lang="python">
@dataclass
class ModelInfo:
    org: str
    base_name: str
    version: str
    size: int
    name: str = None
    is_multimodal: bool = False
    instruct_tag: str = None
    quant_type: QuantType = None
    description: str = None

    @property
    def model_path(self) -> str:
        return f"{self.org}/{self.name}"
</syntaxhighlight>

=== ModelMeta Dataclass ===

<syntaxhighlight lang="python">
@dataclass
class ModelMeta:
    org: str
    base_name: str
    model_version: str
    model_info_cls: type[ModelInfo]
    model_sizes: list[str] = field(default_factory=list)
    instruct_tags: list[str] = field(default_factory=list)
    quant_types: list[QuantType] | dict[str, list[QuantType]] = field(default_factory=list)
    is_multimodal: bool = False
</syntaxhighlight>

== I/O Contract ==

=== QuantType Enum ===

{| class="wikitable"
|-
! Value !! Tag !! Description
|-
| `BNB` || `bnb-4bit` || Standard bitsandbytes 4-bit quantization
|-
| `UNSLOTH` || `unsloth-bnb-4bit` || Unsloth dynamic 4-bit quantization
|-
| `GGUF` || `GGUF` || GGUF format for llama.cpp compatibility
|-
| `BF16` || `bf16` || BFloat16 precision (used for DeepSeek-V3)
|-
| `NONE` || None || Original unquantized model
|}

=== ModelInfo Properties ===

{| class="wikitable"
|-
! Property !! Type !! Description
|-
| `org` || `str` || Organization/namespace (e.g., "unsloth", "deepseek-ai")
|-
| `base_name` || `str` || Base model name (e.g., "Llama", "DeepSeek")
|-
| `version` || `str` || Model version (e.g., "3", "R1")
|-
| `size` || `int` || Model size in billions of parameters
|-
| `name` || `str` || Full constructed model name
|-
| `is_multimodal` || `bool` || Whether model supports multiple modalities
|-
| `instruct_tag` || `str` || Instruction-tuning tag (e.g., "Instruct")
|-
| `quant_type` || `QuantType` || Quantization type applied
|-
| `model_path` || `str` || Full HuggingFace model path (org/name)
|}

=== Core Functions ===

{| class="wikitable"
|-
! Function !! Signature !! Description
|-
| `register_model` || `(model_info_cls, org, base_name, version, size, ...) -> None` || Register a single model variant
|-
| `_register_models` || `(model_meta, include_original_model) -> None` || Bulk register models from metadata
|-
| `_check_model_info` || `(model_id, properties) -> HfModelInfo or None` || Validate model exists on HuggingFace
|}

== Usage Examples ==

=== Define and Register a Model Family ===

<syntaxhighlight lang="python">
from unsloth.registry.registry import (
    ModelInfo, ModelMeta, QuantType, register_model, _register_models
)

# Define custom ModelInfo for naming conventions
class MyModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}-{size}B"
        return super().construct_model_name(
            base_name, version, size, quant_type, instruct_tag, key
        )

# Define model family metadata
MyModelMeta = ModelMeta(
    org="my-org",
    base_name="MyModel",
    model_version="1.0",
    model_sizes=["7", "13"],
    instruct_tags=[None, "Instruct"],
    model_info_cls=MyModelInfo,
    quant_types=[QuantType.UNSLOTH, QuantType.BNB],
)

# Register all variants
_register_models(MyModelMeta, include_original_model=True)
</syntaxhighlight>

=== Register a Single Model ===

<syntaxhighlight lang="python">
from unsloth.registry.registry import register_model, ModelInfo, QuantType

register_model(
    model_info_cls=ModelInfo,
    org="unsloth",
    base_name="Llama",
    version="3",
    size=8,
    instruct_tag="Instruct",
    quant_type=QuantType.UNSLOTH,
)
</syntaxhighlight>

=== Query the Registry ===

<syntaxhighlight lang="python">
from unsloth.registry.registry import MODEL_REGISTRY

# Check if a model is registered
model_id = "unsloth/Llama-3-8B-Instruct-unsloth-bnb-4bit"
if model_id in MODEL_REGISTRY:
    info = MODEL_REGISTRY[model_id]
    print(f"Found: {info.model_path}")
    print(f"Quantization: {info.quant_type}")

# List all registered models
for model_id, info in MODEL_REGISTRY.items():
    print(f"{model_id}: {info.quant_type.value if info.quant_type else 'none'}")
</syntaxhighlight>

=== Validate Model on HuggingFace ===

<syntaxhighlight lang="python">
from unsloth.registry.registry import _check_model_info

# Check if model exists on HuggingFace Hub
model_info = _check_model_info("unsloth/Llama-3.3-70B-Instruct-bnb-4bit")
if model_info:
    print(f"Model found, last modified: {model_info.lastModified}")
else:
    print("Model not found on HuggingFace")
</syntaxhighlight>

=== Per-Size Quantization Types ===

<syntaxhighlight lang="python">
from unsloth.registry.registry import ModelMeta, QuantType

# Different quant types for different sizes
FlexibleModelMeta = ModelMeta(
    org="example",
    base_name="FlexModel",
    model_version="1",
    model_sizes=["7", "70"],
    model_info_cls=ModelInfo,
    quant_types={
        "7": [QuantType.UNSLOTH, QuantType.BNB, QuantType.GGUF],  # Small: all formats
        "70": [QuantType.GGUF],  # Large: only GGUF (memory constraints)
    },
)
</syntaxhighlight>

== Related Pages ==

* [[Unslothai_Unsloth_DeepSeek_Registry|DeepSeek Registry]] - DeepSeek model family registration
* [[Unslothai_Unsloth_Device_Type|Device Type]] - Hardware detection for quantization support
