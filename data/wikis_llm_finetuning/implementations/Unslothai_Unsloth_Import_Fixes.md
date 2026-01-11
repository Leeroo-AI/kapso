# Implementation: Import_Fixes

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Compatibility]], [[domain::Patching]], [[domain::Infrastructure]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Module providing runtime patches and compatibility fixes for various library versions and GPU architectures.

=== Description ===
`import_fixes.py` contains a collection of patches that are applied at import time to fix compatibility issues between different versions of dependencies (transformers, vLLM, TRL, xformers, fbgemm, etc.) and to handle GPU-specific quirks (SM90/SM100 architecture differences, PDL bugs on Blackwell GPUs). It also provides logging configuration and warning suppression.

=== Usage ===
This module is automatically imported by Unsloth's `__init__.py`. Users don't typically interact with it directly, but understanding its functions helps debug compatibility issues.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/import_fixes.py unsloth/import_fixes.py]
* '''Lines:''' 1-696

=== Key Functions ===
<syntaxhighlight lang="python">
def Version(version: str) -> TrueVersion:
    """Parse version string handling dev/alpha/beta/rc suffixes."""

def fix_message_factory_issue() -> None:
    """Patch protobuf MessageFactory for tensorflow compatibility."""

def fix_xformers_performance_issue() -> None:
    """Patch xformers cutlass.py to fix num_splits_key performance bug."""

def fix_vllm_aimv2_issue() -> None:
    """Patch vLLM ovis.py to fix aimv2 config name collision."""

def fix_vllm_guided_decoding_params() -> None:
    """Alias StructuredOutputsParams back to GuidedDecodingParams."""

def patch_enable_input_require_grads() -> None:
    """Patch transformers for vision model NotImplementedError."""

def torchvision_compatibility_check() -> None:
    """Verify torch/torchvision version compatibility."""

def fix_vllm_pdl_blackwell() -> None:
    """Disable PDL (Programmatic Dependent Launch) on SM100 Blackwell GPUs."""

def check_fbgemm_gpu_version() -> None:
    """Check fbgemm_gpu version and disable if too old."""

class HideLoggingMessage(logging.Filter):
    """Filter to hide specific log messages."""

class HidePrintMessage:
    """Stream wrapper to filter stderr messages."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Automatically applied when importing unsloth
import unsloth

# Or import specific fixes
from unsloth.import_fixes import (
    Version,
    fix_vllm_pdl_blackwell,
    torchvision_compatibility_check,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| (automatic) || - || - || Module applies patches on import
|-
| UNSLOTH_ENABLE_LOGGING || env var || No || Set to "1" to enable verbose logging
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Patches applied || Side effects || Various library modules patched in-place
|-
| Warnings suppressed || Side effects || Noisy warnings filtered from stderr
|-
| Compatibility errors || Exception || Raised if incompatible versions detected
|}

== Patches Applied ==

=== Library Version Fixes ===
{| class="wikitable"
|-
! Library !! Condition !! Fix
|-
| protobuf || MessageFactory missing GetPrototype || Add GetPrototype method
|-
| xformers || < 0.0.29 || Fix num_splits_key=-1 to None
|-
| vLLM || < 0.10.1 || Fix aimv2 config name collision
|-
| vLLM || All versions || Alias GuidedDecodingParams
|-
| TRL || OpenEnv 0.26 || Fix SamplingParams not defined
|-
| datasets || 4.4.0 - 4.5.0 || Raise error (recursion bug)
|-
| executorch || Missing torchtune || Add get_mapped_key stub
|}

=== GPU Architecture Fixes ===
{| class="wikitable"
|-
! GPU !! Issue !! Fix
|-
| SM100 (Blackwell) || PDL bug in Triton || Set TRITON_DISABLE_PDL=1, patch supports_pdl()
|-
| fbgemm_gpu < 1.4.0 || Numerical precision || Disable FBGEMM, use Triton kernels
|}

== Usage Examples ==

=== Check Version Compatibility ===
<syntaxhighlight lang="python">
from unsloth.import_fixes import Version

# Compare versions safely
torch_ver = Version("2.5.0")
if torch_ver >= Version("2.4.0"):
    print("Torch 2.4+ features available")
</syntaxhighlight>

=== Enable Verbose Logging ===
<syntaxhighlight lang="bash">
# Set before importing unsloth
export UNSLOTH_ENABLE_LOGGING=1
python your_script.py
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
