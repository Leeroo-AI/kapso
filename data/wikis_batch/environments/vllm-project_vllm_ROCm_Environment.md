# Environment: vllm-project_vllm_ROCm_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM ROCm Docs|https://docs.vllm.ai/en/latest/]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

AMD ROCm environment for GPU-accelerated LLM inference on AMD Instinct GPUs (MI200, MI300 series).

=== Description ===

This environment provides AMD GPU support for vLLM through the ROCm (Radeon Open Compute) stack. It includes specialized AITER (AMD Inference Triton Engine Runtime) operations for optimized performance on MI-series GPUs, with custom paged attention kernels and FP8 support for MI300 cards.

=== Usage ===

Use this environment when deploying on **AMD Instinct GPUs** (MI200, MI250, MI300 series). Supports the same workflows as CUDA but with AMD-specific optimizations including AITER linear ops, custom paged attention, and ROCm-specific quantization.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux || Ubuntu 22.04 recommended
|-
| Hardware || AMD Instinct MI200/MI250/MI300 || Consumer GPUs not officially supported
|-
| ROCm || 5.7+ || Check with `rocm-smi`
|-
| Driver || Compatible with ROCm version || AMDGPU driver required
|-
| Disk || 50GB+ SSD || For model weights and cache
|}

== Dependencies ==

=== System Packages ===

* `rocm` >= 5.7
* `rocm-hip-runtime`
* `rocblas`
* `hipblas`
* `miopen`

=== Python Packages ===

* `torch` (ROCm build)
* `transformers` >= 4.0
* `triton` (ROCm compatible)

== Credentials ==

Same as CUDA environment:
* `HF_TOKEN`: HuggingFace API token
* `VLLM_API_KEY`: API key for vLLM server

== Quick Install ==

<syntaxhighlight lang="bash">
# Install vLLM with ROCm support
pip install vllm

# Set target device explicitly
export VLLM_TARGET_DEVICE=rocm
</syntaxhighlight>

== Code Evidence ==

ROCm detection from `setup.py:578-581`:
<syntaxhighlight lang="python">
def _is_hip() -> bool:
    return (
        VLLM_TARGET_DEVICE == "cuda" or VLLM_TARGET_DEVICE == "rocm"
    ) and torch.version.hip is not None
</syntaxhighlight>

ROCm version detection from `setup.py:600-629`:
<syntaxhighlight lang="python">
def get_rocm_version():
    # Get the Rocm version from the ROCM_HOME/bin/librocm-core.so
    try:
        librocm_core_file = Path(ROCM_HOME) / "lib" / "librocm-core.so"
        if not librocm_core_file.is_file():
            return None
        librocm_core = ctypes.CDLL(librocm_core_file)
        # ...
</syntaxhighlight>

AITER operations control from `vllm/envs.py:929-931`:
<syntaxhighlight lang="python">
"VLLM_ROCM_USE_AITER": lambda: (
    os.getenv("VLLM_ROCM_USE_AITER", "False").lower() in ("true", "1")
),
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ROCM_HOME not found` || ROCm not installed || Install ROCm toolkit
|-
|| `HIP runtime error` || Driver mismatch || Update AMDGPU driver to match ROCm version
|-
|| `torch.version.hip is None` || PyTorch not built for ROCm || Install ROCm-compatible PyTorch
|}

== Compatibility Notes ==

* '''AITER Operations:''' Disabled by default; enable with `VLLM_ROCM_USE_AITER=1` for MI300
* '''FP8 Support:''' Enable with `VLLM_ROCM_FP8_PADDING=1` (default enabled)
* '''Custom Paged Attention:''' Optimized for MI300, enable with `VLLM_ROCM_CUSTOM_PAGED_ATTN=1`
* '''Quick Reduce:''' For large models, use `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8`

== Related Pages ==

* [[requires_env::Implementation:vllm-project_vllm_EngineArgs]]
* [[requires_env::Implementation:vllm-project_vllm_LLM_init]]
