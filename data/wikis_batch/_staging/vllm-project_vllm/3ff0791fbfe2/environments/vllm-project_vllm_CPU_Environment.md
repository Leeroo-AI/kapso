# Environment: vllm-project_vllm_CPU_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Docs|https://docs.vllm.ai/en/latest/]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

CPU-only inference environment for vLLM using Intel IPEX optimizations, suitable for machines without GPU.

=== Description ===

This environment enables vLLM inference on CPU-only systems. It leverages Intel Extension for PyTorch (IPEX) for optimized performance and supports OpenMP thread binding for NUMA-aware execution. While slower than GPU inference, it provides a fallback option for development and testing.

=== Usage ===

Use this environment when **no GPU is available** or for development/testing purposes. Set `VLLM_TARGET_DEVICE=cpu` explicitly or vLLM will auto-detect if CUDA is unavailable.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux or macOS || macOS automatically uses CPU
|-
| Hardware || x86_64 CPU with AVX2+ || Intel Xeon recommended for production
|-
| RAM || 32GB+ || Model-dependent; 7B model needs ~14GB
|-
| Disk || 50GB+ SSD || For model weights
|}

== Dependencies ==

=== System Packages ===

* OpenMP runtime
* Intel MKL (optional, for IPEX)

=== Python Packages ===

* `torch` (CPU build)
* `intel-extension-for-pytorch` (IPEX, optional)
* `transformers` >= 4.0

== Credentials ==

* `HF_TOKEN`: HuggingFace API token for gated model access

== Quick Install ==

<syntaxhighlight lang="bash">
# Set CPU target device
export VLLM_TARGET_DEVICE=cpu

# Install vLLM
pip install vllm
</syntaxhighlight>

== Code Evidence ==

CPU fallback detection from `setup.py:53-61`:
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

CPU KV cache configuration from `vllm/envs.py:690-692`:
<syntaxhighlight lang="python">
# (CPU backend only) CPU key-value cache space.
# default is None and will be set as 4 GB
"VLLM_CPU_KVCACHE_SPACE": lambda: int(os.getenv("VLLM_CPU_KVCACHE_SPACE", "0"))
</syntaxhighlight>

OpenMP thread binding from `vllm/envs.py:695-696`:
<syntaxhighlight lang="python">
# (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31"
"VLLM_CPU_OMP_THREADS_BIND": lambda: os.getenv("VLLM_CPU_OMP_THREADS_BIND", "auto"),
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Out of memory` || Insufficient RAM || Reduce model size or increase swap
|-
|| `IPEX not found` || Intel extensions not installed || Install `intel-extension-for-pytorch`
|-
|| `Slow inference` || Suboptimal thread binding || Set `VLLM_CPU_OMP_THREADS_BIND` for NUMA awareness
|}

== Compatibility Notes ==

* '''Performance:''' Significantly slower than GPU; use for development/testing only
* '''KV Cache:''' Default 4GB; increase with `VLLM_CPU_KVCACHE_SPACE` for larger contexts
* '''Thread Binding:''' Use `VLLM_CPU_OMP_THREADS_BIND="0-31"` for NUMA-aware execution
* '''SGL Kernel:''' Enable `VLLM_CPU_SGL_KERNEL=1` for small batch optimization

== Related Pages ==

* [[requires_env::Implementation:vllm-project_vllm_EngineArgs]]
* [[requires_env::Implementation:vllm-project_vllm_LLM_init]]
