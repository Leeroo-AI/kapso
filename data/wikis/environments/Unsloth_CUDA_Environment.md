{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
* [[source::Doc|NVIDIA CUDA Toolkit|https://developer.nvidia.com/cuda-toolkit]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Linux environment with CUDA 11.8+/12.x, Python 3.10+, and PyTorch 2.0+ optimized for Unsloth's 2x faster LLM fine-tuning.

=== Description ===
This environment provides the GPU-accelerated context required for running Unsloth's optimized fine-tuning workflows. It is built around the NVIDIA CUDA toolkit and includes all necessary dependencies for parameter-efficient fine-tuning (PEFT) of large language models. The environment supports both Ampere (A100, RTX 30xx) and newer architectures (H100, RTX 40xx), with specific optimizations for memory-efficient training through gradient checkpointing and 4-bit quantization.

=== Usage ===
Use this environment for any **Model Fine-Tuning**, **Reinforcement Learning from Human Feedback (RLHF)**, or **Inference** workflow using Unsloth. It is the mandatory prerequisite for running `FastLanguageModel`, `FastModel`, and all Unsloth-optimized training implementations. Required when you need the 2x speedup and 70% VRAM reduction that Unsloth provides.

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
|| OS || Linux (Ubuntu 20.04/22.04 LTS) || WSL2 on Windows also supported
|-
|| Hardware || NVIDIA GPU || Minimum 8GB VRAM (RTX 3060+, A10, T4); 24GB+ recommended for 7B+ models
|-
|| CUDA || 11.8 or 12.x || Must match PyTorch CUDA version
|-
|| Disk || 50GB SSD || For model weights and checkpoints
|-
|| RAM || 16GB+ || 32GB recommended for larger models
|}

== Dependencies ==
=== System Packages ===
* `cuda-toolkit` >= 11.8
* `cudnn` >= 8.6
* `git`
* `git-lfs`

=== Python Packages ===
* `python` >= 3.10
* `torch` >= 2.0.0
* `unsloth` (latest)
* `transformers` >= 4.36.0
* `trl` >= 0.7.0
* `peft` >= 0.7.0
* `bitsandbytes` >= 0.41.0
* `accelerate` >= 0.25.0
* `datasets`
* `xformers` (optional, for memory efficiency)

== Credentials ==
The following environment variables may be required:
* `HF_TOKEN`: HuggingFace API token for gated models (Llama, Mistral, etc.)
* `WANDB_API_KEY`: Weights & Biases API key for experiment tracking (optional)

== Related Pages ==
=== Required By ===
* [[required_by::Implementation:Unsloth_FastLanguageModel]]
* [[required_by::Implementation:Unsloth_FastModel]]
* [[required_by::Implementation:TRL_SFTTrainer]]
* [[required_by::Implementation:TRL_DPOTrainer]]
* [[required_by::Implementation:TRL_GRPOTrainer]]
* [[required_by::Implementation:Unsloth_get_peft_model]]
* [[required_by::Implementation:BitsAndBytes_4bit_Quantization]]
* [[required_by::Implementation:Unsloth_save_pretrained]]
* [[required_by::Implementation:Unsloth_save_to_GGUF]]

