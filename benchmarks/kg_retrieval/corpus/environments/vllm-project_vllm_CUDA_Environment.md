# Environment: CUDA_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM installation|https://docs.vllm.ai/en/latest/getting_started/installation/]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Linux environment with an NVIDIA GPU, a driver supporting the CUDA runtime that the vLLM wheel was built against, and a matching PyTorch build.

=== Description ===

vLLM ships compiled attention kernels tied to a specific CUDA and PyTorch combination. Installing it into an environment with a different torch usually replaces torch and can break other packages.

=== Usage ===

Create a clean virtual environment dedicated to serving.

== System Requirements ==

* NVIDIA GPU of compute capability 7.5 or higher
* Driver new enough for the wheel's CUDA runtime
* Python 3.11 or newer

== Quick Install ==

<syntaxhighlight lang="bash">
python -m venv serve-env && . serve-env/bin/activate
pip install vllm
</syntaxhighlight>
