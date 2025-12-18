# Python Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Requirements|https://docs.vllm.ai/en/latest/getting_started/installation.html]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Python]]
|-
! Last Updated
| [[last_updated::2025-01-15 14:00 GMT]]
|}

== Overview ==

Python 3.9+ environment with core vLLM dependencies for CPU-side operations including tokenization, sampling parameter configuration, and output processing.

=== Description ===

This environment provides the standard Python runtime required for vLLM operations that do not directly require GPU acceleration. It includes tokenization libraries (HuggingFace Transformers), serialization tools (msgspec, pydantic), and the core vLLM Python API. This environment is used for parameter configuration, prompt formatting, and output processing operations.

=== Usage ===

Use this environment for **CPU-side operations** that do not require GPU access:
- `SamplingParams` configuration
- `PromptType` formatting (text prompts, token IDs)
- `RequestOutput` processing and text extraction
- `LoRARequest` object creation
- `StructuredOutputsParams` configuration
- Multimodal data preparation (image loading via PIL)

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, macOS, Windows (via WSL2) || Linux recommended for production
|-
| Python || Python 3.9 - 3.12 || Python 3.10+ recommended
|-
| Memory || 8GB+ RAM || More for large tokenizers/vocabularies
|-
| Disk || 10GB+ || For package installation and model tokenizers
|}

== Dependencies ==

=== Python Packages ===
* `transformers` >= 4.40.0 (tokenization, model configs)
* `tokenizers` >= 0.15.0 (fast tokenizer backend)
* `msgspec` >= 0.18.0 (fast serialization for SamplingParams)
* `pydantic` >= 2.0.0 (data validation)
* `pillow` >= 9.0.0 (image processing for VLMs)
* `numpy` >= 1.21.0
* `tqdm` (progress bars)
* `cloudpickle` (function serialization)

=== Optional Dependencies ===
* `outlines` (structured output generation backend)
* `lm-format-enforcer` (alternative structured output backend)
* `xgrammar` (grammar-based generation)

== Credentials ==

The following environment variables may be used:
* `HF_TOKEN`: HuggingFace API token for downloading tokenizers from gated models

== Quick Install ==

<syntaxhighlight lang="bash">
# Install core Python dependencies
pip install transformers tokenizers msgspec pydantic pillow numpy tqdm

# For structured output support
pip install outlines lm-format-enforcer

# Verify installation
python -c "from vllm import SamplingParams; print('SamplingParams available')"
</syntaxhighlight>

== Code Evidence ==

SamplingParams uses msgspec for efficient serialization (`vllm/sampling_params.py:111-117`):
<syntaxhighlight lang="python">
class SamplingParams(
    PydanticMsgspecMixin,
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
):
    """Sampling parameters for text generation."""
</syntaxhighlight>

StructuredOutputsParams validation (`vllm/sampling_params.py:52-68`):
<syntaxhighlight lang="python">
def __post_init__(self):
    """Validate that some fields are mutually exclusive."""
    count = sum([
        self.json is not None,
        self.regex is not None,
        self.choice is not None,
        self.grammar is not None,
        self.json_object is not None,
        self.structural_tag is not None,
    ])
    if count > 1:
        raise ValueError(
            "You can only use one kind of structured outputs constraint "
            f"but multiple are specified: {self.__dict__}"
        )
</syntaxhighlight>

LoRARequest uses msgspec (`vllm/lora/request.py:9-15`):
<syntaxhighlight lang="python">
class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,
    array_like=True,
):
    """Request for a LoRA adapter."""
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: No module named 'transformers'` || transformers not installed || `pip install transformers`
|-
|| `ImportError: No module named 'msgspec'` || msgspec not installed || `pip install msgspec`
|-
|| `ValidationError: You can only use one kind of structured outputs` || Multiple structured output types specified || Use only one of json/regex/choice/grammar
|-
|| `ValueError: temperature must be >= 0` || Invalid sampling parameter || Set temperature to non-negative value
|-
|| `TokenizerError: Tokenizer not found` || Missing tokenizer files || Ensure model path is correct and HF_TOKEN is set for gated models
|}

== Compatibility Notes ==

* '''Python 3.9:''' Minimum supported version, full functionality.
* '''Python 3.10-3.12:''' Recommended for best performance with newer typing features.
* '''Python 3.13+:''' Not yet tested; may have compatibility issues.
* '''msgspec vs pydantic:''' vLLM uses msgspec for performance-critical paths; pydantic for validation.
* '''Tokenizer modes:''' "auto" uses fast tokenizer when available; "slow" forces HuggingFace slow tokenizer.

== Related Pages ==

* [[requires_env::Implementation:vllm-project_vllm_SamplingParams_init]]
* [[requires_env::Implementation:vllm-project_vllm_PromptType_usage]]
* [[requires_env::Implementation:vllm-project_vllm_RequestOutput_usage]]
* [[requires_env::Implementation:vllm-project_vllm_LoRARequest_init]]
* [[requires_env::Implementation:vllm-project_vllm_RequestOutput_lora]]
* [[requires_env::Implementation:vllm-project_vllm_MultiModalData_image]]
* [[requires_env::Implementation:vllm-project_vllm_VLM_prompt_format]]
* [[requires_env::Implementation:vllm-project_vllm_RequestOutput_vlm]]
* [[requires_env::Implementation:vllm-project_vllm_SpeculativeMethod_choice]]
* [[requires_env::Implementation:vllm-project_vllm_SpeculativeConfig_init]]
* [[requires_env::Implementation:vllm-project_vllm_get_metrics_spec]]
* [[requires_env::Implementation:vllm-project_vllm_StructuredOutputsParams_types]]
* [[requires_env::Implementation:vllm-project_vllm_StructuredOutputsParams_init]]
* [[requires_env::Implementation:vllm-project_vllm_SamplingParams_structured]]
* [[requires_env::Implementation:vllm-project_vllm_structured_output_parse]]
