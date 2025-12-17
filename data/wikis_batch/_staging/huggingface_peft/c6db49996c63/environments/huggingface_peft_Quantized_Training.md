# Environment: huggingface_peft_Quantized_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Repo|bitsandbytes|https://github.com/bitsandbytes-foundation/bitsandbytes]]
* [[source::Doc|QLoRA Paper|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Quantization]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
CUDA environment with bitsandbytes for 4-bit/8-bit quantized model training (QLoRA).

=== Description ===
This environment extends the base CUDA training environment with quantization support via bitsandbytes. It enables loading large language models in 4-bit or 8-bit precision, significantly reducing memory requirements while maintaining model quality. PEFT provides native integration with multiple quantization backends including bitsandbytes, GPTQ, AWQ, AQLM, EETQ, HQQ, and TorchAO.

=== Usage ===
Use this environment for **QLoRA training** workflows where memory constraints prevent loading full-precision models. Essential for fine-tuning 7B+ parameter models on consumer GPUs (16-24GB VRAM). Also required for inference with quantized base models.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+) || Windows requires WSL2; bitsandbytes has limited Windows support
|-
| Hardware || NVIDIA GPU (CUDA 11.0+) || Ampere (A100) or newer recommended for optimal 4-bit performance
|-
| VRAM || 8GB+ minimum || 16GB+ recommended for 7B models; 24GB+ for 13B models
|-
| Python || >= 3.10.0 || Required by PEFT
|}

== Dependencies ==

=== System Packages ===
* `CUDA toolkit` >= 11.0 (11.8 or 12.x recommended)
* `cuDNN` (matching CUDA version)

=== Python Packages ===
* All packages from `huggingface_peft_CUDA_Training` environment
* `bitsandbytes` (for 4-bit/8-bit quantization)

=== Optional Quantization Backends ===
* `auto-gptq` >= 0.5.0 (for GPTQ quantization)
* `gptqmodel` >= 2.0.0 (requires `optimum` >= 1.24.0)
* `awq` (for AWQ quantization)
* `aqlm` (for AQLM quantization)
* `eetq` (for EETQ quantization)
* `hqq` (for HQQ quantization)
* `torchao` >= 0.4.0 (for PyTorch AO quantization)

== Credentials ==
The following environment variables may be required:
* `HF_TOKEN`: HuggingFace API token (for gated models like Llama)
* `HF_HOME`: Optional - HuggingFace cache directory

== Quick Install ==
<syntaxhighlight lang="bash">
# Install PEFT with bitsandbytes support
pip install peft bitsandbytes

# For GPTQ support
pip install peft auto-gptq

# For all quantization backends
pip install peft bitsandbytes auto-gptq awq aqlm eetq hqq torchao
</syntaxhighlight>

== Code Evidence ==

Quantization backend detection from `src/peft/import_utils.py:24-36`:
<syntaxhighlight lang="python">
@lru_cache
def is_bnb_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None

@lru_cache
def is_bnb_4bit_available() -> bool:
    if not is_bnb_available():
        return False
    import bitsandbytes as bnb
    return hasattr(bnb.nn, "Linear4bit")
</syntaxhighlight>

GPTQ version check from `src/peft/import_utils.py:39-49`:
<syntaxhighlight lang="python">
@lru_cache
def is_auto_gptq_available():
    if importlib.util.find_spec("auto_gptq") is not None:
        AUTOGPTQ_MINIMUM_VERSION = packaging.version.parse("0.5.0")
        version_autogptq = packaging.version.parse(importlib_metadata.version("auto_gptq"))
        if AUTOGPTQ_MINIMUM_VERSION <= version_autogptq:
            return True
        else:
            raise ImportError(
                f"Found an incompatible version of auto-gptq. Found version {version_autogptq}, "
                f"but only versions above {AUTOGPTQ_MINIMUM_VERSION} are supported"
            )
</syntaxhighlight>

TorchAO version validation from `src/peft/import_utils.py:127-147`:
<syntaxhighlight lang="python">
@lru_cache
def is_torchao_available():
    if importlib.util.find_spec("torchao") is None:
        return False
    TORCHAO_MINIMUM_VERSION = packaging.version.parse("0.4.0")
    torchao_version = packaging.version.parse(importlib_metadata.version("torchao"))
    if torchao_version < TORCHAO_MINIMUM_VERSION:
        raise ImportError(
            f"Found an incompatible version of torchao. Found version {torchao_version}, "
            f"but only versions above {TORCHAO_MINIMUM_VERSION} are supported"
        )
    return True
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: No module named 'bitsandbytes'` || bitsandbytes not installed || `pip install bitsandbytes`
|-
|| `CUDA Setup failed despite GPU being available` || CUDA/bitsandbytes version mismatch || Reinstall bitsandbytes matching your CUDA version
|-
|| `Found an incompatible version of auto-gptq` || auto-gptq < 0.5.0 || `pip install --upgrade auto-gptq>=0.5.0`
|-
|| `Found an incompatible version of torchao` || torchao < 0.4.0 || `pip install --upgrade torchao>=0.4.0`
|-
|| `gptqmodel requires optimum version` || optimum too old for gptqmodel || `pip install --upgrade optimum>=1.24.0`
|}

== Compatibility Notes ==

* **bitsandbytes on Windows:** Requires CUDA 12.1+ and has limited support; use WSL2 for full compatibility
* **AMD GPUs (ROCm):** bitsandbytes has experimental ROCm support; check latest release notes
* **Apple Silicon:** bitsandbytes not supported; use standard precision or other backends
* **Multi-GPU:** 4-bit models work with `device_map="auto"` for model parallelism
* **Nested Quantization:** Use `bnb_4bit_use_double_quant=True` for additional memory savings

== Related Pages ==
* [[requires_env::Implementation:huggingface_peft_BitsAndBytesConfig]]
* [[requires_env::Implementation:huggingface_peft_prepare_model_for_kbit_training]]

[[Category:Environment]]
[[Category:Infrastructure]]
[[Category:Quantization]]
[[Category:QLoRA]]
