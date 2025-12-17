# Environment: vllm-project_vllm_CUDA_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Docs|https://docs.vllm.ai/en/latest/]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

NVIDIA CUDA environment with CUDA 11.8+ or 12.x, Python 3.10-3.13, and PyTorch 2.9.0 for GPU-accelerated LLM inference.

=== Description ===

This environment provides the primary GPU-accelerated context for vLLM inference. It is built on top of NVIDIA CUDA and includes support for advanced features like FlashAttention, PagedAttention, and quantized inference (FP8, INT8, AWQ, GPTQ). The environment automatically detects GPU capabilities and selects optimal kernels for Ampere (A100), Hopper (H100), and Blackwell architectures.

=== Usage ===

Use this environment for **all GPU-based inference** workflows including offline batch inference, online API serving, LoRA adapter inference, speculative decoding, and vision-language multimodal inference. It is the mandatory prerequisite for running the core vLLM implementations.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (including WSL) or macOS || Windows not officially supported; macOS defaults to CPU
|-
| Hardware || NVIDIA GPU with Compute Capability 7.0+ || A100/H100 recommended; minimum 16GB VRAM for 7B models
|-
| CUDA || 11.8+ or 12.x || CUDA 12.3+ required for FlashAttention 3
|-
| Driver || Compatible with CUDA version || Check `nvidia-smi` for driver version
|-
| Disk || 50GB+ SSD || For model weights and cache
|}

== Dependencies ==

=== System Packages ===

* `cuda-toolkit` >= 11.8
* `cudnn` >= 8.0
* `nccl` >= 2.18 (for multi-GPU)
* `cmake` >= 3.26.1
* `ninja`
* `gcc` (for CUDA compilation)

=== Python Packages ===

* `torch` == 2.9.0
* `transformers` >= 4.0
* `triton` (for Triton kernels)
* `flashinfer` (optional, for FlashInfer backend)
* `vllm-flash-attn` (for FlashAttention, CUDA 12.x only)
* `msgspec`
* `pydantic`

== Credentials ==

The following environment variables may be required:
* `HF_TOKEN`: HuggingFace API token for gated model access
* `VLLM_API_KEY`: API key for vLLM server authentication
* `S3_ACCESS_KEY_ID`: For S3-based model loading with tensorizer
* `S3_SECRET_ACCESS_KEY`: For S3-based model loading
* `S3_ENDPOINT_URL`: Custom S3 endpoint URL

== Quick Install ==

<syntaxhighlight lang="bash">
# Install vLLM with CUDA support
pip install vllm

# Or install from source for development
pip install torch==2.9.0
pip install -e .
</syntaxhighlight>

== Code Evidence ==

Device detection from `setup.py:53-61`:
<syntaxhighlight lang="python">
elif (
    sys.platform.startswith("linux")
    and torch.version.cuda is None
    and os.getenv("VLLM_TARGET_DEVICE") is None
    and torch.version.hip is None
):
    # if cuda or hip is not available and VLLM_TARGET_DEVICE is not set,
    # fallback to cpu
    VLLM_TARGET_DEVICE = "cpu"
</syntaxhighlight>

CUDA version check from `setup.py:632-644`:
<syntaxhighlight lang="python">
def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc."""
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output(
        [CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version
</syntaxhighlight>

FlashAttention 3 CUDA version check from `setup.py:748-750`:
<syntaxhighlight lang="python">
if envs.VLLM_USE_PRECOMPILED or get_nvcc_cuda_version() >= Version("12.3"):
    # FA3 requires CUDA 12.3 or later
    ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `CUDA_HOME is not set` || CUDA toolkit not found || Set `CUDA_HOME` to CUDA installation path (e.g., `/usr/local/cuda`)
|-
|| `Cannot find CMake executable` || CMake not installed || `pip install cmake>=3.26.1`
|-
|| `RuntimeError: Unknown runtime environment` || Unsupported platform || Use Linux with CUDA, ROCm, or CPU target
|-
|| `CUDA out of memory` || Insufficient GPU VRAM || Reduce `gpu_memory_utilization` or use smaller model
|-
|| `vllm-flash-attn not found` || FlashAttention not installed for CUDA 12.x || `pip install vllm-flash-attn`
|}

== Compatibility Notes ==

* '''macOS:''' Automatically defaults to CPU mode; GPU not supported
* '''Windows:''' Not officially supported; use WSL2 for GPU acceleration
* '''CUDA < 12.3:''' FlashAttention 3 not available; FA2 will be used
* '''Compute Capability < 7.0:''' Not supported; requires Volta or newer
* '''Multi-GPU:''' Requires NCCL for tensor parallelism; check P2P connectivity with `VLLM_SKIP_P2P_CHECK=0`

== Related Pages ==

* [[requires_env::Implementation:vllm-project_vllm_EngineArgs]]
* [[requires_env::Implementation:vllm-project_vllm_LLM_init]]
* [[requires_env::Implementation:vllm-project_vllm_LLM_generate]]
* [[requires_env::Implementation:vllm-project_vllm_SamplingParams]]
* [[requires_env::Implementation:vllm-project_vllm_vllm_serve]]
