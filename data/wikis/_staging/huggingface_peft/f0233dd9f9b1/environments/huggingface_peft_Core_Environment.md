# Environment: huggingface_peft_Core_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Installation|https://huggingface.co/docs/peft/install]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]], [[domain::Parameter_Efficient_Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2024-12-18 00:00 GMT]]
|}

== Overview ==
Python 3.10+ environment with PyTorch 1.13+, HuggingFace Transformers, and Accelerate for PEFT adapter training.

=== Description ===
This environment provides the core dependencies for Parameter-Efficient Fine-Tuning (PEFT). It includes the base requirements for loading pre-trained models, configuring LoRA/other adapters, and saving trained adapters. This is the minimum viable environment for using PEFT without any quantization or specialized initialization methods.

=== Usage ===
Use this environment for any basic PEFT workflow including LoRA fine-tuning, adapter loading, inference, and multi-adapter management. This environment is sufficient when:
- Training adapters on full-precision models (fp16/bf16/fp32)
- Loading and merging adapters for inference
- Managing multiple adapters on a single model

For quantized models (QLoRA), see `huggingface_peft_Quantization_Environment`.

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, macOS, Windows || Linux recommended for GPU training
|-
| Python || >= 3.10.0 || Required by PEFT
|-
| Hardware || CPU or CUDA GPU || GPU recommended for training
|-
| Disk || 10GB+ || For model weights and checkpoints
|}

== Dependencies ==
=== System Packages ===
* No special system packages required for basic usage
* CUDA toolkit if using NVIDIA GPU

=== Python Packages ===
* `torch` >= 1.13.0
* `transformers` (any recent version)
* `accelerate` >= 0.21.0
* `safetensors` (any recent version)
* `huggingface_hub` >= 0.25.0
* `numpy` >= 1.17
* `packaging` >= 20.0
* `pyyaml`
* `tqdm`
* `psutil`

== Credentials ==
The following environment variables are optional but useful:
* `HF_TOKEN`: HuggingFace API token for accessing gated models (e.g., Llama)
* `HF_HUB_OFFLINE`: Set to "1" to enable offline mode

== Quick Install ==
<syntaxhighlight lang="bash">
# Install PEFT with all core dependencies
pip install peft

# Or install from source
pip install git+https://github.com/huggingface/peft.git

# Verify installation
python -c "import peft; print(peft.__version__)"
</syntaxhighlight>

== Code Evidence ==

Core dependency requirements from `setup.py:60-71`:
<syntaxhighlight lang="python">
install_requires=[
    "numpy>=1.17",
    "packaging>=20.0",
    "psutil",
    "pyyaml",
    "torch>=1.13.0",
    "transformers",
    "tqdm",
    "accelerate>=0.21.0",
    "safetensors",
    "huggingface_hub>=0.25.0",
],
</syntaxhighlight>

Python version requirement from `setup.py:59`:
<syntaxhighlight lang="python">
python_requires=">=3.10.0",
</syntaxhighlight>

Device inference logic from `other.py:116-127`:
<syntaxhighlight lang="python">
def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif mlu_available:
        return "mlu"
    elif is_xpu_available():
        return "xpu"
    elif is_npu_available():
        return "npu"
    return "cpu"
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ModuleNotFoundError: No module named 'peft'` || PEFT not installed || `pip install peft`
|-
|| `ImportError: accelerate` || Accelerate not installed or too old || `pip install accelerate>=0.21.0`
|-
|| `RuntimeError: CUDA out of memory` || Insufficient GPU VRAM || Use gradient checkpointing or smaller batch size
|-
|| `ValueError: You have to provide either input_ids or inputs_embeds` || Missing input to model || Ensure proper tokenization
|}

== Compatibility Notes ==

* **macOS MPS**: Apple Silicon supported via MPS backend
* **Intel XPU**: Supported via PyTorch XPU integration
* **Huawei NPU**: Supported via Ascend integration with accelerate >= 0.29.0
* **Cambricon MLU**: Supported via MLU integration with accelerate >= 0.29.0
* **CPU-only**: Training possible but significantly slower

== Related Pages ==
* [[requires_env::Implementation:huggingface_peft_LoraConfig_init]]
* [[requires_env::Implementation:huggingface_peft_get_peft_model]]
* [[requires_env::Implementation:huggingface_peft_PeftModel_from_pretrained]]
* [[requires_env::Implementation:huggingface_peft_PeftModel_save_pretrained]]
* [[requires_env::Implementation:huggingface_peft_merge_and_unload]]
