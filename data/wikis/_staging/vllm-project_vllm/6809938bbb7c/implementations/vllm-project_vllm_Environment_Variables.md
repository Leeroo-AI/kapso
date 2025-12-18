{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Configuration]], [[domain::Environment]], [[domain::System]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Centralized environment variable management system for vLLM configuration and runtime behavior control.

=== Description ===
The envs.py module provides a comprehensive environment variable management system for vLLM, exposing over 200 configurable settings that control various aspects of the inference engine's behavior. This module serves as the single source of truth for all environment-based configuration, including device selection, optimization flags, distributed training settings, logging configuration, and feature toggles.

The module uses type annotations and helper functions (env_with_choices, env_list_with_choices, env_set_with_choices) to provide validation for environment variables, ensuring that only valid values are accepted. It includes settings for CUDA/ROCm GPU operations, CPU optimizations, memory management, model loading, attention backends, quantization methods, and numerous debugging flags. All environment variables follow a consistent VLLM_ prefix naming convention.

The configuration system supports dynamic defaults based on platform detection and torch version, with many settings having conditional behavior. This centralized approach allows users to customize vLLM's behavior without modifying code, making it essential for deployment flexibility and debugging.

=== Usage ===
Import this module to access environment variable values throughout the vLLM codebase. Users set environment variables before launching vLLM to configure behavior.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/envs.py vllm/envs.py]
* '''Lines:''' 1-1750

=== Signature ===
<syntaxhighlight lang="python">
# Key configuration categories
VLLM_TARGET_DEVICE: str = "cuda"
VLLM_LOGGING_LEVEL: str = "INFO"
VLLM_ATTENTION_BACKEND: str | None = None
VLLM_USE_PRECOMPILED: bool = False
VLLM_WORKER_MULTIPROC_METHOD: Literal["fork", "spawn"] = "fork"

# Environment validation helpers
def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]

def env_list_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], list[str]]

def env_set_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], set[str]]

# Cache and config paths
def get_default_cache_root() -> str
def get_default_config_root() -> str

# Specialized helpers
def use_aot_compile() -> bool
def disable_compile_cache() -> bool
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
import vllm.envs as envs

# Access environment variables
device = envs.VLLM_TARGET_DEVICE
log_level = envs.VLLM_LOGGING_LEVEL
use_fp8 = envs.VLLM_USE_TRITON_AWQ
</syntaxhighlight>

== I/O Contract ==

=== Key Exports ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| VLLM_TARGET_DEVICE || str || Target device: cuda, cpu, xpu, rocm, neuron, tpu, etc.
|-
| VLLM_LOGGING_LEVEL || str || Logging level: DEBUG, INFO, WARNING, ERROR
|-
| VLLM_ATTENTION_BACKEND || str | None || Attention backend selection
|-
| VLLM_CACHE_ROOT || str || Root directory for cached models and data
|-
| VLLM_CONFIG_ROOT || str || Root directory for configuration files
|-
| VLLM_USE_PRECOMPILED || bool || Use precompiled kernels if available
|-
| VLLM_ROCM_USE_AITER || bool || Enable AITER optimizations on ROCm
|-
| VLLM_WORKER_MULTIPROC_METHOD || Literal["fork", "spawn"] || Multiprocessing method
|-
| VLLM_DISABLE_FLASHINFER_PREFILL || bool || Disable flashinfer for prefill
|-
| VLLM_DISABLED_KERNELS || list[str] || List of disabled custom kernels
|-
| env_with_choices || Callable || Validator for single-value env vars
|-
| env_list_with_choices || Callable || Validator for comma-separated env vars
|-
| get_default_cache_root || Callable || Get default cache directory
|-
| use_aot_compile || Callable || Check if AOT compilation is enabled
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Basic usage - reading environment variables
import vllm.envs as envs

# Check device configuration
if envs.VLLM_TARGET_DEVICE == "rocm":
    print("Running on ROCm")
    if envs.VLLM_ROCM_USE_AITER:
        print("AITER optimizations enabled")

# Configure logging
print(f"Log level: {envs.VLLM_LOGGING_LEVEL}")
print(f"Log to: {envs.VLLM_LOGGING_STREAM}")

# Check cache paths
cache_root = envs.VLLM_CACHE_ROOT
config_root = envs.VLLM_CONFIG_ROOT
print(f"Cache: {cache_root}, Config: {config_root}")

# Check compilation settings
if envs.use_aot_compile():
    print("AOT compilation enabled")

# Check disabled kernels
if "fused_moe" in envs.VLLM_DISABLED_KERNELS:
    print("Fused MoE kernel is disabled")

# Using validation helpers
from vllm.envs import env_with_choices

# Define a validated environment variable
get_backend = env_with_choices(
    "MY_BACKEND",
    default="flash_attn",
    choices=["flash_attn", "xformers", "triton"],
    case_sensitive=False
)

backend = get_backend()  # Will validate against choices
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
