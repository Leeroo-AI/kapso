# Environment: CUDA_GPU_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|device_type.py|https://github.com/unslothai/unsloth/blob/main/unsloth/device_type.py]]
* [[source::Doc|loader.py|https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::LLMs]], [[domain::Infrastructure]]
|-
! Last Updated
| [[last_updated::2026-01-09 12:00 GMT]]
|}

== Overview ==
GPU-accelerated environment supporting NVIDIA CUDA, AMD ROCm (HIP), and Intel XPU for training and fine-tuning LLMs with Unsloth.

=== Description ===
This environment provides the core GPU acceleration context for all Unsloth training workflows. It supports three GPU backends:

1. **NVIDIA CUDA** - Primary support with full feature set
2. **AMD ROCm (HIP)** - Support via bitsandbytes >= 0.48.3, with some limitations on pre-quantized models
3. **Intel XPU** - Requires PyTorch >= 2.6.0

The environment auto-detects available GPU accelerators and configures the appropriate compute backend. BFloat16 support is automatically detected based on GPU compute capability.

=== Usage ===
Use this environment for **QLoRA fine-tuning**, **Vision fine-tuning**, and **GGUF export** workflows. This is the standard prerequisite for running `FastLanguageModel.from_pretrained()`, `FastVisionModel.from_pretrained()`, and all model saving operations.

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+) || Windows via WSL2 only
|-
| Hardware || NVIDIA/AMD/Intel GPU || Minimum 16GB VRAM recommended for 7B models
|-
| CUDA || >= 11.8 || For NVIDIA GPUs
|-
| ROCm || >= 6.0 || For AMD GPUs
|-
| Disk || 50GB SSD || For model caching and checkpoints
|}

== Dependencies ==
=== System Packages ===
* `cuda-toolkit` >= 11.8 (NVIDIA)
* `rocm` >= 6.0 (AMD)
* `intel-extension-for-pytorch` (Intel XPU)

=== Python Packages ===
* `torch` >= 2.4.0
* `transformers` >= 4.45.0 (4.37+ for 4-bit support, 4.43.2+ for Llama 3.1)
* `bitsandbytes` >= 0.43.3 (0.48.3+ for AMD ROCm)
* `peft` >= 0.10.0
* `trl` >= 0.11.0
* `triton` >= 3.0.0
* `accelerate`
* `datasets`
* `huggingface_hub`

== Credentials ==
The following environment variables may be required:
* `HF_TOKEN`: HuggingFace API token for private model access and Hub uploads
* `WANDB_API_KEY`: Weights & Biases API key for training logging (optional)

== Quick Install ==
<syntaxhighlight lang="bash">
# Install all required packages
pip install torch>=2.4.0 transformers>=4.45.0 bitsandbytes>=0.43.3 peft>=0.10.0 trl>=0.11.0 accelerate triton>=3.0.0 datasets

# For Unsloth
pip install unsloth
</syntaxhighlight>

== Code Evidence ==

Device detection from `device_type.py:37-59`:
<syntaxhighlight lang="python">
@functools.cache
def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if is_hip():
            return "hip"
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    # Check torch.accelerator
    if hasattr(torch, "accelerator"):
        if not torch.accelerator.is_available():
            raise NotImplementedError(
                "Unsloth cannot find any torch accelerator? You need a GPU."
            )
    raise NotImplementedError(
        "Unsloth currently only works on NVIDIA, AMD and Intel GPUs."
    )
</syntaxhighlight>

Intel XPU version check from `kernels/utils.py:41-44`:
<syntaxhighlight lang="python">
if DEVICE_TYPE == "xpu" and Version(torch.__version__) < Version("2.6.0"):
    raise RuntimeError(
        "Intel xpu currently supports unsloth with torch.version >= 2.6.0"
    )
</syntaxhighlight>

Transformers version checks from `loader.py:68-79`:
<syntaxhighlight lang="python">
SUPPORTS_FOURBIT = transformers_version >= Version("4.37")
SUPPORTS_GEMMA = transformers_version >= Version("4.38")
SUPPORTS_GEMMA2 = transformers_version >= Version("4.42")
SUPPORTS_LLAMA31 = transformers_version >= Version("4.43.2")
SUPPORTS_LLAMA32 = transformers_version > Version("4.45.0")
SUPPORTS_GRANITE = transformers_version >= Version("4.46.0")
SUPPORTS_QWEN3 = transformers_version >= Version("4.50.3")
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Unsloth cannot find any torch accelerator? You need a GPU.` || No GPU detected || Ensure CUDA/ROCm drivers are installed and GPU is available
|-
|| `Intel xpu currently supports unsloth with torch.version >= 2.6.0` || Old PyTorch for Intel || Upgrade: `pip install torch>=2.6.0`
|-
|| `transformers version does not support Llama 3.1` || transformers < 4.43.2 || Upgrade: `pip install transformers>=4.43.2`
|-
|| `AMD currently is not stable with 4bit bitsandbytes` || BnB < 0.48.3 on AMD || Upgrade: `pip install bitsandbytes>=0.48.3`
|-
|| `Device does not support bfloat16` || Older GPU (pre-Ampere) || Unsloth auto-switches to float16
|}

== Compatibility Notes ==

* **AMD GPUs (ROCm):** Requires bitsandbytes >= 0.48.3. Pre-quantized models with blocksize 64 may not work on Instinct GPUs (MI series) which use blocksize 128. Radeon (Navi) GPUs are supported.
* **Intel XPU:** Requires PyTorch >= 2.6.0. Uses `torch.xpu` device type.
* **Windows:** Not officially supported; use WSL2.
* **BFloat16:** Auto-detected based on GPU capability. Falls back to float16 on older GPUs.
* **Multi-GPU:** Supported via device_map="sequential". DEVICE_COUNT is auto-detected.

== Related Pages ==
* [[required_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained]]
* [[required_by::Implementation:Unslothai_Unsloth_get_peft_model]]
* [[required_by::Implementation:Unslothai_Unsloth_get_chat_template]]
* [[required_by::Implementation:Unslothai_Unsloth_UnslothTrainingArguments]]
* [[required_by::Implementation:Unslothai_Unsloth_SFTTrainer_train]]
* [[required_by::Implementation:Unslothai_Unsloth_save_pretrained]]
* [[required_by::Implementation:Unslothai_Unsloth_FastVisionModel_from_pretrained]]
* [[required_by::Implementation:Unslothai_Unsloth_FastBaseModel_get_peft_model]]
* [[required_by::Implementation:Unslothai_Unsloth_multimodal_dataset_pattern]]
* [[required_by::Implementation:Unslothai_Unsloth_FastBaseModel_for_training]]
* [[required_by::Implementation:Unslothai_Unsloth_SFTTrainer_vision]]
* [[required_by::Implementation:Unslothai_Unsloth_FastBaseModel_for_inference]]
* [[required_by::Implementation:Unslothai_Unsloth_save_pretrained_vision]]
* [[required_by::Implementation:Unslothai_Unsloth_unsloth_save_model_merged]]
* [[required_by::Implementation:Unslothai_Unsloth_ALLOWED_QUANTS]]
* [[required_by::Implementation:Unslothai_Unsloth_save_to_gguf]]
* [[required_by::Implementation:Unslothai_Unsloth_OLLAMA_TEMPLATES]]
* [[required_by::Implementation:Unslothai_Unsloth_push_to_hub_gguf]]
