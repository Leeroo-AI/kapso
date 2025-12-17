# Environment: huggingface_peft_CUDA_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Installation|https://huggingface.co/docs/peft/install]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]], [[domain::Parameter_Efficient_Finetuning]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
Standard CUDA-enabled Python environment for training PEFT adapters with PyTorch 1.13+ and transformers.

=== Description ===
This environment provides the baseline context for running PEFT (Parameter-Efficient Fine-Tuning) operations. It supports GPU-accelerated training using PyTorch with CUDA, along with the HuggingFace transformers ecosystem. PEFT automatically detects available hardware including CUDA, XPU (Intel), NPU, MLU, and MPS (Apple Silicon), defaulting to CPU when no accelerator is available.

=== Usage ===
Use this environment for any **LoRA fine-tuning**, **adapter inference**, or **multi-adapter management** workflow that requires GPU acceleration. This is the prerequisite environment for all PEFT training and inference operations.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, Windows, macOS || Python 3.10+ required
|-
| Hardware || CUDA-capable GPU recommended || Falls back to CPU; also supports Intel XPU, Apple MPS
|-
| Disk || 5GB+ || For model caching and checkpoints
|-
| Python || >= 3.10.0 || Required by setup.py
|}

== Dependencies ==

=== System Packages ===
* `CUDA toolkit` (11.x or 12.x recommended for PyTorch 2.x)
* `git` (for HuggingFace Hub operations)

=== Python Packages ===
* `torch` >= 1.13.0
* `transformers` (latest recommended)
* `accelerate` >= 0.21.0
* `safetensors` (for safe model serialization)
* `huggingface_hub` >= 0.25.0
* `numpy` >= 1.17
* `pyyaml`
* `tqdm`
* `psutil`
* `packaging` >= 20.0

== Credentials ==
The following environment variables may be required:
* `HF_TOKEN`: HuggingFace API token (for private models/repos)
* `HF_HOME`: Optional - HuggingFace cache directory override

== Quick Install ==
<syntaxhighlight lang="bash">
# Install PEFT with all core dependencies
pip install peft

# Or install from source
pip install git+https://github.com/huggingface/peft.git

# For development/testing
pip install peft[dev,test]
</syntaxhighlight>

== Code Evidence ==

Device auto-detection from `src/peft/utils/other.py:116-127`:
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

Dependencies from `setup.py:60-71`:
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

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `RuntimeError: CUDA out of memory` || Insufficient GPU VRAM || Reduce batch size, enable gradient checkpointing, or use QLoRA
|-
|| `ImportError: No module named 'transformers'` || Missing transformers || `pip install transformers`
|-
|| `ValueError: The following `model_kwargs` are not used` || Model kwargs mismatch || Check device_map and torch_dtype compatibility
|}

== Compatibility Notes ==

* **Apple Silicon (MPS):** Supported but may have limited functionality for some operations
* **Intel XPU:** Requires `torch.xpu` availability (PyTorch with Intel extension)
* **Windows:** Full support but WSL2 recommended for best experience
* **CPU-only:** Fully functional but significantly slower for training

== Related Pages ==
* [[requires_env::Implementation:huggingface_peft_get_peft_model]]
* [[requires_env::Implementation:huggingface_peft_LoraConfig]]
* [[requires_env::Implementation:huggingface_peft_save_pretrained]]
* [[requires_env::Implementation:huggingface_peft_PeftModel_from_pretrained]]
* [[requires_env::Implementation:huggingface_peft_merge_and_unload]]
* [[requires_env::Implementation:huggingface_peft_load_adapter]]
* [[requires_env::Implementation:huggingface_peft_set_adapter]]

[[Category:Environment]]
[[Category:Infrastructure]]
[[Category:PEFT]]
