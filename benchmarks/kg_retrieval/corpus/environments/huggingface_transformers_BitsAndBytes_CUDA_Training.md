# Environment: BitsAndBytes_CUDA_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|bitsandbytes|https://github.com/bitsandbytes-foundation/bitsandbytes]]
* [[source::Doc|bitsandbytes installation|https://huggingface.co/docs/bitsandbytes/installation]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Linux environment with an NVIDIA GPU, a CUDA runtime matching the installed bitsandbytes wheel, and compatible torch and transformers versions for 4-bit model loading and training.

=== Description ===

The 4-bit kernels live in a compiled shared library selected at import time by the detected CUDA runtime. A missing or mismatched library silently disables GPU support or raises at load.

=== Usage ===

Verify it before launching adapter training on a compressed base model.

== System Requirements ==

* NVIDIA GPU of compute capability 6.0 or newer for 4-bit layers (7.5 or newer for 8-bit matrix multiplication)
* CUDA runtime visible to the process

== Dependencies ==

* bitsandbytes, torch, transformers and accelerate at mutually compatible versions

== Quick Install ==

<syntaxhighlight lang="bash">
pip install bitsandbytes transformers accelerate
python -m bitsandbytes
</syntaxhighlight>

== Common Errors ==

* ''Method ... not available in CPU-only version of bitsandbytes'': only the CPU library was loaded, so GPU layers cannot be used.
* ''CUDA runtime libraries were not found in the LD_LIBRARY_PATH'': the runtime directory is missing from the library path.
* ''CUDA VERSION MISMATCH'': the wheel contains no binary for the CUDA version PyTorch was built with.
