# GPU Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Installation|https://docs.vllm.ai/en/latest/getting_started/installation.html]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]], [[domain::GPU_Computing]]
|-
! Last Updated
| [[last_updated::2025-01-15 14:00 GMT]]
|}

== Overview ==

Linux environment with NVIDIA CUDA 11.8+ (or ROCm for AMD GPUs), Python 3.9+, and PyTorch 2.0+ for GPU-accelerated LLM inference.

=== Description ===

This environment provides GPU-accelerated context for running vLLM's high-throughput inference engine. It supports NVIDIA GPUs via CUDA (11.8 or higher) and AMD GPUs via ROCm. The environment includes optimized attention kernels (FlashAttention, FlashInfer), quantization backends (AWQ, GPTQ, FP8), and tensor parallelism for multi-GPU inference. It is optimized for Ampere (A100), Hopper (H100), and Ada Lovelace (L40S, RTX 4090) architectures.

=== Usage ===

Use this environment for any **LLM inference** operation that requires GPU acceleration. It is the mandatory prerequisite for running the `LLM` class, `LLM.generate()` method, and any GPU-bound implementations including:
- Model initialization and weight loading
- Batch generation and continuous batching
- Speculative decoding
- LoRA adapter serving
- Vision-language model inference

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04/22.04 recommended) || Windows not officially supported; use WSL2
|-
| Hardware || NVIDIA GPU with compute capability >= 7.0 || Minimum 16GB VRAM recommended (A100/H100 preferred)
|-
| CUDA || CUDA 11.8 or CUDA 12.x || CUDA 12.1+ recommended for optimal performance
|-
| ROCm || ROCm 6.0+ (AMD only) || For MI250X/MI300X GPUs
|-
| Memory || 32GB+ system RAM || Required for model loading and CPU offload
|-
| Disk || 50GB+ SSD || High IOPS required for model weights and KV cache spilling
|}

== Dependencies ==

=== System Packages ===
* `cuda-toolkit` >= 11.8 (NVIDIA)
* `cudnn` >= 8.6 (NVIDIA)
* `nccl` >= 2.18 (multi-GPU communication)
* `cmake` >= 3.21
* `git-lfs` (for model downloads)

=== Python Packages ===
* `torch` >= 2.0.0
* `vllm` (this package)
* `transformers` >= 4.40.0
* `triton` >= 2.2.0 (for Triton kernels)
* `xformers` (optional, for memory-efficient attention)
* `flash-attn` >= 2.5.0 (optional, for FlashAttention)
* `flashinfer` (optional, for FlashInfer backend)

== Credentials ==

The following environment variables may be required:
* `HF_TOKEN`: HuggingFace API token for gated model downloads (e.g., Llama models)
* `CUDA_VISIBLE_DEVICES`: Control which GPUs are visible to the process

== Quick Install ==

<syntaxhighlight lang="bash">
# Install vLLM with CUDA support
pip install vllm

# For specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
</syntaxhighlight>

== Code Evidence ==

Device detection from `vllm/platforms/__init__.py`:
<syntaxhighlight lang="python">
# Platform auto-detection based on available accelerators
if torch.cuda.is_available():
    platform_cls = CudaPlatform
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    platform_cls = XPUPlatform
elif hasattr(torch.version, "hip") and torch.version.hip is not None:
    platform_cls = RocmPlatform
</syntaxhighlight>

CUDA version check from `setup.py:111-112`:
<syntaxhighlight lang="python">
if _is_cuda() and get_nvcc_cuda_version() >= Version("11.2"):
    # nvcc_threads is supported for parallel compilation
</syntaxhighlight>

Target device configuration from `vllm/envs.py:456`:
<syntaxhighlight lang="python">
# Target device of vLLM, supporting [cuda (by default), rocm, cpu]
"VLLM_TARGET_DEVICE": lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda").lower(),
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `CUDA out of memory` || Insufficient GPU VRAM for model + KV cache || Reduce `gpu_memory_utilization` (default 0.9) to 0.7-0.8, or use quantization
|-
|| `RuntimeError: CUDA error: no kernel image is available` || GPU compute capability not supported || Verify GPU has compute capability >= 7.0 (Volta or newer)
|-
|| `torch.cuda.is_available() returns False` || CUDA drivers not installed or misconfigured || Install NVIDIA drivers and CUDA toolkit
|-
|| `ImportError: libcudart.so not found` || CUDA runtime library missing || Set `LD_LIBRARY_PATH` to include CUDA lib directory
|-
|| `NCCL error: unhandled system error` || Multi-GPU communication failure || Check GPU interconnect (NVLink/PCIe), try `NCCL_DEBUG=INFO`
|}

== Compatibility Notes ==

* '''NVIDIA GPUs:''' Recommended for best performance. Supports Volta (V100), Ampere (A100, A10), Hopper (H100), Ada (L40S, RTX 4090).
* '''AMD GPUs (ROCm):''' Supported via ROCm 6.0+. Some features may have different performance characteristics.
* '''Intel XPU:''' Experimental support via IPEX backend.
* '''macOS:''' Not supported for GPU inference; CPU-only mode available.
* '''Windows:''' Not officially supported; use WSL2 with Ubuntu.
* '''Tensor Parallelism:''' Requires same GPU type across all devices.

== Related Pages ==

* [[requires_env::Implementation:vllm-project_vllm_LLM_init]]
* [[requires_env::Implementation:vllm-project_vllm_LLM_generate]]
* [[requires_env::Implementation:vllm-project_vllm_EngineArgs_lora]]
* [[requires_env::Implementation:vllm-project_vllm_LLMEngine_add_request_lora]]
* [[requires_env::Implementation:vllm-project_vllm_Scheduler_lora_batching]]
* [[requires_env::Implementation:vllm-project_vllm_EngineArgs_vlm]]
* [[requires_env::Implementation:vllm-project_vllm_LLM_generate_mm]]
* [[requires_env::Implementation:vllm-project_vllm_LLM_speculative]]
* [[requires_env::Implementation:vllm-project_vllm_LLM_generate_spec]]
* [[requires_env::Implementation:vllm-project_vllm_LLM_generate_structured]]
