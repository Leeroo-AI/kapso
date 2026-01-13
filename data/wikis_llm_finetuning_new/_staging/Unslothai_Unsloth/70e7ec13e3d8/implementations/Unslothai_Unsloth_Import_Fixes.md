# Implementation: Import_Fixes

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Library_Compatibility]], [[domain::System_Integration]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==
Import_fixes provides a collection of runtime patches and compatibility fixes for various library conflicts and version mismatches in the deep learning ecosystem.

=== Description ===
This module contains multiple patch functions that address compatibility issues between Unsloth and its dependencies. The fixes are applied at import time to ensure smooth operation across different library versions:

'''Protobuf Fixes''': fix_message_factory_issue() patches google.protobuf.MessageFactory to add missing GetPrototype method, resolving AttributeError issues caused by tensorflow/protobuf version conflicts.

'''Xformers Performance''': fix_xformers_performance_issue() patches xformers < 0.0.29 to fix num_splits_key parameter causing performance degradation.

'''vLLM Compatibility''':
* fix_vllm_aimv2_issue() patches vLLM < 0.10.1 to avoid "aimv2 is already used by a Transformers config" error
* fix_vllm_guided_decoding_params() aliases GuidedDecodingParams to StructuredOutputsParams for TRL compatibility
* fix_vllm_pdl_blackwell() disables PDL (Programmatic Dependent Launch) on SM100 Blackwell GPUs to avoid Triton CUDA graph capture failures

'''HuggingFace Patches''':
* fix_huggingface_hub() restores removed is_offline_mode() function
* patch_ipykernel_hf_xet() works around hf_xet + ipykernel progress bar crashes

'''TRL/Transformers Fixes''':
* fix_openenv_no_vllm() patches TRL OpenEnv to handle missing vLLM gracefully
* patch_enable_input_require_grads() fixes vision model compatibility with enable_input_require_grads

'''Other Utilities''':
* Version() - robust version string parser with dev/alpha/beta handling
* HideLoggingMessage / HidePrintMessage - filter classes for suppressing noisy log messages
* torchvision_compatibility_check() - validates torch/torchvision version compatibility
* check_fbgemm_gpu_version() - validates FBGEMM GPU version and disables if problematic

=== Usage ===
This module is typically imported automatically by Unsloth's main entry points. Import directly if you need to apply specific patches manually or check compatibility before loading other modules.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' unsloth/import_fixes.py
* '''Lines:''' 1-695

=== Signature ===
<syntaxhighlight lang="python">
# Version utilities
def Version(version: str) -> packaging.version.Version: ...

# Logging filters
class HideLoggingMessage(logging.Filter):
    def __init__(self, text: str) -> None: ...
    def filter(self, x: logging.LogRecord) -> bool: ...

class HidePrintMessage:
    def __init__(self, original_stream) -> None: ...
    def add_filter(self, text: str) -> None: ...
    def write(self, message: str) -> None: ...

# Patch functions
def fix_message_factory_issue() -> None: ...
def fix_xformers_performance_issue() -> None: ...
def fix_vllm_aimv2_issue() -> None: ...
def fix_vllm_guided_decoding_params() -> None: ...
def fix_vllm_pdl_blackwell() -> None: ...
def fix_openenv_no_vllm() -> None: ...
def fix_executorch() -> None: ...
def fix_diffusers_warnings() -> None: ...
def fix_huggingface_hub() -> None: ...
def ignore_logger_messages() -> None: ...
def patch_ipykernel_hf_xet() -> None: ...
def patch_trackio() -> None: ...
def patch_datasets() -> None: ...
def patch_enable_input_require_grads() -> None: ...

# Compatibility checks
def check_fbgemm_gpu_version() -> None: ...
def torchvision_compatibility_check() -> None: ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.import_fixes import (
    Version,
    fix_message_factory_issue,
    fix_xformers_performance_issue,
    fix_vllm_aimv2_issue,
    fix_vllm_pdl_blackwell,
    fix_huggingface_hub,
    torchvision_compatibility_check,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| (Version) version || str || Yes || Version string to parse (e.g., "0.0.29", "2.1.0.dev1")
|-
| (HideLoggingMessage) text || str || Yes || Substring to filter from log messages
|-
| (HidePrintMessage) original_stream || TextIO || Yes || Original stdout/stderr stream to wrap
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Version return || packaging.version.Version || Parsed version object for comparison
|-
| All patch functions || None || Functions modify libraries in-place; no return value
|}

== Usage Examples ==
<syntaxhighlight lang="python">
from unsloth.import_fixes import (
    Version,
    fix_message_factory_issue,
    fix_xformers_performance_issue,
    fix_vllm_pdl_blackwell,
    torchvision_compatibility_check,
    HideLoggingMessage,
)
import logging

# Compare library versions robustly
xformers_version = "0.0.28.post1"
if Version(xformers_version) < Version("0.0.29"):
    print("Xformers needs patching")
    fix_xformers_performance_issue()

# Apply protobuf fix before importing tensorflow
fix_message_factory_issue()
import tensorflow  # Now safe to import

# Apply vLLM PDL fix for Blackwell GPUs before using LoRA
fix_vllm_pdl_blackwell()

# Validate torch/torchvision compatibility at startup
try:
    torchvision_compatibility_check()
except ImportError as e:
    print(f"Version mismatch: {e}")

# Suppress specific log messages
logger = logging.getLogger("transformers")
logger.addFilter(HideLoggingMessage("Some weights of"))

# Apply all fixes at once (typical usage pattern)
from unsloth.import_fixes import (
    fix_message_factory_issue,
    fix_xformers_performance_issue,
    fix_vllm_aimv2_issue,
    fix_huggingface_hub,
    ignore_logger_messages,
)

# Apply patches in order
fix_message_factory_issue()
fix_xformers_performance_issue()
fix_vllm_aimv2_issue()
fix_huggingface_hub()
ignore_logger_messages()

# Now safe to import main libraries
from transformers import AutoModelForCausalLM
from unsloth import FastLanguageModel
</syntaxhighlight>

== Related Pages ==
* [[related::Implementation:Unslothai_Unsloth_Device_Type]]
* [[requires_env::Environment:Unslothai_Unsloth_TRL]]
* [[requires_env::Environment:Unslothai_Unsloth_PEFT]]
