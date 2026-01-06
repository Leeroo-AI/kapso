{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Module_System]], [[domain::Package_Management]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
The main __init__.py file implements a lazy loading system that defers heavy imports until explicitly requested, enabling fast "import transformers" without loading PyTorch or other large dependencies.

=== Description ===
This file serves as the entry point for the entire Transformers library and implements a sophisticated lazy import mechanism using _LazyModule. The import structure is defined declaratively in _import_structure dictionary, mapping submodules to their exported names. This allows the package to provide all names in the namespace immediately while deferring actual imports until attributes are accessed.

The file handles optional dependencies gracefully by checking availability (is_torch_available(), is_tokenizers_available(), etc.) and substituting dummy objects when dependencies are missing. It includes a TYPE_CHECKING branch that provides proper type hints for IDEs and static analyzers without triggering actual imports. The lazy loading system significantly improves import performance - "import transformers" completes in milliseconds instead of seconds by avoiding PyTorch initialization.

The file also creates tokenization aliases for backwards compatibility, redirecting legacy module paths to their replacements without importing the target modules. Additionally, it emits a warning if PyTorch is not available, informing users that only tokenizers and utilities will work.

=== Usage ===
This file is automatically executed when users run "import transformers" or "from transformers import ...". Developers adding new classes must update both _import_structure (for lazy loading) and the TYPE_CHECKING section (for type hints). The lazy system works transparently - users don't need to know it exists.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' src/transformers/__init__.py

=== Signature ===
<syntaxhighlight lang="python">
# Core lazy module setup
_import_structure = {
    "configuration_utils": ["PreTrainedConfig", "PretrainedConfig"],
    "modeling_utils": ["PreTrainedModel", "AttentionInterface"],
    "generation": ["GenerationConfig", "GenerationMixin", ...],
    # ... hundreds of modules and classes
}

# Optional dependency handling
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects
    _import_structure["utils.dummy_pt_objects"] = [...]
else:
    _import_structure["modeling_utils"] = ["PreTrainedModel", ...]

# Lazy module activation
sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    import_structure,
    module_spec=__spec__,
    extra_objects={"__version__": __version__}
)

# Backwards compatibility aliases
def _create_tokenization_alias(alias: str, target: str) -> None:
    # Creates lazy alias modules without importing
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Basic import (fast, no heavy dependencies loaded)
import transformers

# Lazy attribute access (triggers import on first access)
from transformers import AutoModel  # Loads modeling code now
from transformers import pipeline    # Loads pipeline code now

# Check version (available immediately, no lazy loading)
print(transformers.__version__)  # "5.0.0.dev0"

# Check dependency availability (imported eagerly)
from transformers import is_torch_available
if is_torch_available():
    from transformers import Trainer
</syntaxhighlight>

== I/O Contract ==

=== Key Module Categories in _import_structure ===
{| class="wikitable"
|-
! Category !! Modules !! Description
|-
| Configuration || configuration_utils, PreTrainedConfig || Model configuration classes
|-
| Tokenization || tokenization_python, tokenization_utils_base, tokenization_utils_fast || Tokenizer implementations
|-
| Modeling || modeling_utils, PreTrainedModel, modeling_outputs || Core model classes
|-
| Generation || generation.*, GenerationConfig, LogitsProcessor || Text generation utilities
|-
| Data || data.*, DataCollator, datasets || Data processing and collation
|-
| Training || trainer, Trainer, TrainingArguments, optimization || Training infrastructure
|-
| Pipelines || pipelines.*, pipeline, Pipeline || High-level API for inference
|-
| Utils || utils.*, logging, import_utils || Shared utilities
|}

=== Optional Dependency Branches ===
{| class="wikitable"
|-
! Check !! Modules Loaded !! Fallback
|-
| is_torch_available() || modeling_utils, trainer, optimization, activations || dummy_pt_objects
|-
| is_tokenizers_available() || tokenization_utils_tokenizers, PreTrainedTokenizerFast || dummy_tokenizers_objects
|-
| is_vision_available() || image_processing_base, image_utils || dummy_vision_objects
|-
| is_torchvision_available() || image_processing_utils_fast, video_processing_utils || dummy_torchvision_objects
|-
| is_sentencepiece_available() + is_tokenizers_available() || convert_slow_tokenizer || dummy_sentencepiece_and_tokenizers_objects
|-
| is_mistral_common_available() || tokenization_mistral_common || dummy_mistral_common_objects
|}

=== Exported Constants ===
{| class="wikitable"
|-
! Constant !! Value !! Purpose
|-
| __version__ || "5.0.0.dev0" || Package version string
|-
| logging || transformers.utils.logging || Logging utilities
|-
| is_torch_available || Function || Check PyTorch availability
|-
| is_tokenizers_available || Function || Check tokenizers availability
|-
| is_vision_available || Function || Check Pillow availability
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Fast import - no heavy dependencies loaded
import transformers
print(f"Loaded transformers {transformers.__version__}")  # Instant

# Lazy loading - import happens on attribute access
from transformers import AutoModel  # Triggers PyTorch import here
from transformers import AutoTokenizer  # Separate lazy import

# Check what's available before importing
from transformers import is_torch_available
if is_torch_available():
    from transformers import Trainer, TrainingArguments
else:
    print("PyTorch not available, training not supported")

# Pipeline access (triggers pipeline + model imports)
from transformers import pipeline
classifier = pipeline("sentiment-analysis")

# Type checking works correctly (TYPE_CHECKING branch)
from transformers import PreTrainedModel, PreTrainedTokenizer
def process_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    # IDE gets full type hints without importing
    pass

# Backwards compatibility aliases work transparently
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
# Actually imports from tokenization_utils_tokenizers

# Access internal import structure (advanced usage)
import sys
lazy_mod = sys.modules['transformers']
print(lazy_mod._import_structure.keys())  # See all available modules
</syntaxhighlight>

== Related Pages ==
* (Leave empty)
