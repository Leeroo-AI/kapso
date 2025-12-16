# Implementation: Model Registry

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Hub|https://huggingface.co/docs/huggingface_hub/]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Model_Management]], [[domain::Registry]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
Core registry infrastructure for tracking and managing LLM models with their quantization variants, sizes, and metadata in a centralized dictionary.

=== Description ===
The Model Registry system provides a centralized mechanism for registering, tracking, and discovering LLM model variants supported by Unsloth. It defines data structures and registration functions that enable systematic management of hundreds of model variants while maintaining metadata consistency.

Key components:

1. **QuantType Enum** - Defines supported quantization formats:
   - `BNB` - BitsAndBytes 4-bit quantization
   - `UNSLOTH` - Unsloth's dynamic 4-bit quantization
   - `GGUF` - GGUF format for llama.cpp
   - `NONE` - Unquantized models
   - `BF16` - BFloat16 format (for specific models like DeepSeek V3)

2. **ModelInfo Dataclass** - Represents individual model variants with:
   - Organization, base name, version, size
   - Quantization type and multimodal flag
   - `model_path` property for HuggingFace path construction

3. **ModelMeta Dataclass** - Template for model families:
   - Defines all sizes and quantization types for a model family
   - Supports size-specific quantization constraints

4. **Registration Functions**:
   - `register_model()` - Add single model to registry with duplicate checking
   - `_register_models()` - Batch register all variants from a ModelMeta template
   - `_check_model_info()` - Validate against HuggingFace Hub

The separation of `ModelInfo` (instances) and `ModelMeta` (templates) provides clean abstraction for model discovery and management.

=== Usage ===
Import these components when extending Unsloth's model support or programmatically discovering available models. Model definitions in `unsloth/registry/` (e.g., `_llama.py`, `_qwen.py`) use these primitives.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/registry/registry.py#L1-L191 unsloth/registry/registry.py]
* '''Lines:''' 1-191 (full module)

=== Signature ===
<syntaxhighlight lang="python">
class QuantType(Enum):
    """Supported quantization types for models."""
    BNB = "bnb"        # BitsAndBytes 4-bit
    UNSLOTH = "unsloth"  # Dynamic 4-bit quantization
    GGUF = "GGUF"      # GGUF format
    NONE = "none"      # Unquantized
    BF16 = "bf16"      # BFloat16 (DeepSeek V3)


@dataclass
class ModelInfo:
    """
    Represents a single model variant with full metadata.

    Attributes:
        org: Organization/namespace (e.g., "unsloth", "meta-llama")
        base_name: Base model name (e.g., "Llama-3.1")
        version: Model version string
        size: Parameter count (e.g., 8 for 8B)
        name: Full constructed model name
        is_multimodal: Whether model supports images
        instruct_tag: Instruct variant tag (e.g., "Instruct")
        quant_type: Quantization format
    """
    org: str
    base_name: str
    version: str
    size: int
    name: str = None
    is_multimodal: bool = False
    instruct_tag: str = None
    quant_type: QuantType = None

    @property
    def model_path(self) -> str:
        """Returns HuggingFace model path: '{org}/{name}'"""


@dataclass
class ModelMeta:
    """
    Template for registering model families with multiple variants.

    Attributes:
        org: Original organization
        base_name: Base model name
        model_version: Version string
        model_info_cls: ModelInfo subclass to use
        model_sizes: List of supported sizes
        instruct_tags: Variant tags to register
        quant_types: Supported quantizations (list or dict per size)
        is_multimodal: Whether models support images
    """


def register_model(
    model_info_cls: type[ModelInfo],
    org: str,
    base_name: str,
    version: str,
    size: int,
    instruct_tag: str = None,
    quant_type: QuantType = None,
    is_multimodal: bool = False,
    name: str = None,
) -> None:
    """Register a single model variant to MODEL_REGISTRY."""


def _register_models(
    model_meta: ModelMeta,
    include_original_model: bool = False
) -> None:
    """Batch register all variants from a ModelMeta template."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.registry import MODEL_REGISTRY
from unsloth.registry.registry import (
    QuantType,
    ModelInfo,
    ModelMeta,
    register_model,
    _register_models,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_info_cls || type[ModelInfo] || Yes || ModelInfo class or subclass to instantiate
|-
| org || str || Yes || Organization namespace (e.g., "unsloth")
|-
| base_name || str || Yes || Base model name (e.g., "Llama-3.1")
|-
| version || str || Yes || Model version string
|-
| size || int || Yes || Parameter count (e.g., 8 for 8B)
|-
| quant_type || QuantType || No || Quantization format (default: NONE)
|-
| is_multimodal || bool || No || Whether model is multimodal (default: False)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| MODEL_REGISTRY || dict[str, ModelInfo] || Global registry mapping model paths to ModelInfo objects
|-
| model_path || str || HuggingFace model path property (e.g., "unsloth/Llama-3.1-8B-Instruct-bnb-4bit")
|}

== Usage Examples ==

=== Querying the Registry ===
<syntaxhighlight lang="python">
from unsloth.registry import MODEL_REGISTRY

# List all registered models
print(f"Total registered models: {len(MODEL_REGISTRY)}")

# Find specific model
if "unsloth/Llama-3.1-8B-Instruct-bnb-4bit" in MODEL_REGISTRY:
    model_info = MODEL_REGISTRY["unsloth/Llama-3.1-8B-Instruct-bnb-4bit"]
    print(f"Size: {model_info.size}B")
    print(f"Quantization: {model_info.quant_type}")
    print(f"Multimodal: {model_info.is_multimodal}")

# Filter by quantization type
from unsloth.registry.registry import QuantType
bnb_models = [k for k, v in MODEL_REGISTRY.items()
              if v.quant_type == QuantType.BNB]
print(f"BNB quantized models: {len(bnb_models)}")
</syntaxhighlight>

=== Registering a New Model Family ===
<syntaxhighlight lang="python">
from dataclasses import dataclass
from unsloth.registry.registry import (
    ModelInfo, ModelMeta, QuantType, _register_models
)

# Define custom ModelInfo for new architecture
@dataclass
class MyModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag, key=""):
        # Custom naming convention
        key = f"{base_name}-{version}-{size}B"
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key

# Define model family metadata
my_model_meta = ModelMeta(
    org="my-org",
    base_name="MyModel",
    model_version="1.0",
    model_info_cls=MyModelInfo,
    model_sizes=["7", "13", "70"],
    instruct_tags=["Instruct", "Chat"],
    quant_types=[QuantType.BNB, QuantType.UNSLOTH],
    is_multimodal=False,
)

# Register all variants
_register_models(my_model_meta, include_original_model=True)
</syntaxhighlight>

=== Validating Against HuggingFace ===
<syntaxhighlight lang="python">
from unsloth.registry.registry import _check_model_info

# Verify model exists on HuggingFace Hub
model_id = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit"
info = _check_model_info(model_id, properties=["lastModified", "downloads"])

if info:
    print(f"Model found: {info.id}")
    print(f"Last modified: {info.lastModified}")
else:
    print("Model not found on HuggingFace Hub")
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:unslothai_unsloth_GPU_CUDA_Environment]]

=== Tips and Tricks ===
