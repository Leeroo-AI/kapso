# Implementation: Model_Registry

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Models]], [[domain::Registry]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Model registration system for tracking supported models with their quantization variants.

=== Description ===
This module provides a registry system for Unsloth-supported models. It defines dataclasses for model metadata (`ModelInfo`, `ModelMeta`), quantization types (`QuantType`), and functions to register and look up models. The registry maps model paths to their metadata including organization, version, size, and quantization type.

=== Usage ===
Used internally to manage the catalog of supported models and their pre-quantized variants on the Unsloth HuggingFace organization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/registry/registry.py unsloth/registry/registry.py]
* '''Lines:''' 1-191

=== Key Classes ===
<syntaxhighlight lang="python">
class QuantType(Enum):
    """Quantization type enum."""
    BNB = "bnb"           # bitsandbytes 4-bit
    UNSLOTH = "unsloth"   # Dynamic 4-bit
    GGUF = "GGUF"         # GGUF format
    NONE = "none"         # No quantization
    BF16 = "bf16"         # BF16 (DeepSeek V3)

@dataclass
class ModelInfo:
    """Information about a specific model variant."""
    org: str              # Organization (e.g., "unsloth", "meta-llama")
    base_name: str        # Base model name
    version: str          # Version string
    size: int             # Model size (e.g., 7, 8, 70)
    name: str = None      # Full model name
    is_multimodal: bool = False
    instruct_tag: str = None
    quant_type: QuantType = None

@dataclass
class ModelMeta:
    """Metadata for registering model families."""
    org: str
    base_name: str
    model_version: str
    model_info_cls: type[ModelInfo]
    model_sizes: list[str]
    instruct_tags: list[str]
    quant_types: list[QuantType]
    is_multimodal: bool = False
</syntaxhighlight>

=== Key Functions ===
<syntaxhighlight lang="python">
def register_model(
    model_info_cls: ModelInfo,
    org: str,
    base_name: str,
    version: str,
    size: int,
    instruct_tag: str = None,
    quant_type: QuantType = None,
    is_multimodal: bool = False,
    name: str = None,
) -> None:
    """Register a model in MODEL_REGISTRY."""

MODEL_REGISTRY: dict[str, ModelInfo]
    """Global registry mapping model paths to ModelInfo."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.registry.registry import (
    MODEL_REGISTRY,
    ModelInfo,
    ModelMeta,
    QuantType,
    register_model,
)
</syntaxhighlight>

== Quantization Tags ==

{| class="wikitable"
|-
! QuantType !! HF Tag !! Example Path
|-
| BNB || bnb-4bit || unsloth/llama-3-8b-bnb-4bit
|-
| UNSLOTH || unsloth-bnb-4bit || unsloth/llama-3-8b-unsloth-bnb-4bit
|-
| GGUF || GGUF || unsloth/llama-3-8b-GGUF
|-
| BF16 || bf16 || unsloth/DeepSeek-V3-bf16
|-
| NONE || (none) || meta-llama/Llama-3-8B
|}

== Usage Examples ==

=== Check if Model is Registered ===
<syntaxhighlight lang="python">
from unsloth.registry.registry import MODEL_REGISTRY

model_path = "unsloth/llama-3-8b-bnb-4bit"
if model_path in MODEL_REGISTRY:
    info = MODEL_REGISTRY[model_path]
    print(f"Size: {info.size}B, Quant: {info.quant_type}")
</syntaxhighlight>

=== Register Custom Model Family ===
<syntaxhighlight lang="python">
from unsloth.registry.registry import (
    ModelMeta, ModelInfo, QuantType, _register_models
)

custom_meta = ModelMeta(
    org="my-org",
    base_name="my-model",
    model_version="v1",
    model_info_cls=ModelInfo,
    model_sizes=["7B", "13B"],
    instruct_tags=["Instruct"],
    quant_types=[QuantType.BNB, QuantType.NONE],
)
_register_models(custom_meta)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
