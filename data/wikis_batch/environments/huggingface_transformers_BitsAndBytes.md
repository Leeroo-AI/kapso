# Environment: huggingface_transformers_BitsAndBytes

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Quantization Guide|https://huggingface.co/docs/transformers/quantization]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Quantization]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Environment for INT8/INT4 quantization using the bitsandbytes library for memory-efficient model loading.

=== Description ===

This environment enables loading large language models with significantly reduced memory footprint using bitsandbytes quantization. It supports both 8-bit (INT8) and 4-bit (NF4/FP4) quantization methods, allowing models that would normally require 40GB+ VRAM to run on consumer GPUs with 16-24GB. The quantization happens on-the-fly during model loading, with minimal accuracy loss.

=== Usage ===

Use this environment when:
- Loading models too large for available GPU memory
- Running **7B+ parameter models** on consumer GPUs (RTX 3090/4090)
- Performing **QLoRA fine-tuning** (4-bit base model + LoRA adapters)
- Deploying models with **memory constraints**

Required when using `BitsAndBytesConfig` with `load_in_8bit=True` or `load_in_4bit=True`.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux || Windows has limited support
|-
| Hardware || NVIDIA GPU || CUDA-capable GPU required
|-
| VRAM || 8GB+ || 4-bit allows larger models on smaller GPUs
|-
| CUDA || 11.0+ || Must match bitsandbytes build
|-
| Compute Capability || 7.0+ || Volta architecture or newer
|}

== Dependencies ==

=== Inherits from CUDA Environment ===

All dependencies from `huggingface_transformers_CUDA`, plus:

=== Required ===

* `bitsandbytes` >= 0.43.0 (check BITSANDBYTES_MIN_VERSION)
* `accelerate` >= 1.1.0 (required for device mapping)
* `scipy` (for certain quantization operations)

=== Optional for Training ===

* `peft` >= 0.18.0 (for QLoRA fine-tuning)
* `trl` (for RLHF with quantized models)

== Credentials ==

Same as CUDA environment. No additional credentials required.

== Quick Install ==

<syntaxhighlight lang="bash">
# Install bitsandbytes with CUDA support
pip install bitsandbytes>=0.43.0

# Install transformers and accelerate
pip install transformers accelerate

# Verify installation
python -c "import bitsandbytes; print(f'BnB version: {bitsandbytes.__version__}')"
</syntaxhighlight>

== Code Evidence ==

Environment validation from `quantizer_bnb_8bit.py:54-66`:

<syntaxhighlight lang="python">
def validate_environment(self, *args, **kwargs):
    if not is_accelerate_available():
        raise ImportError(
            f"Using `bitsandbytes` 8-bit quantization requires accelerate: "
            f"`pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
        )
    if not is_bitsandbytes_available():
        raise ImportError(
            f"Using `bitsandbytes` 8-bit quantization requires bitsandbytes: "
            f"`pip install -U bitsandbytes>={BITSANDBYTES_MIN_VERSION}`"
        )

    from ..integrations import validate_bnb_backend_availability
    validate_bnb_backend_availability(raise_exception=True)
</syntaxhighlight>

Memory adjustment heuristic from `quantizer_bnb_8bit.py:81-84`:

<syntaxhighlight lang="python">
def adjust_max_memory(self, max_memory: dict[str, int | str]) -> dict[str, int | str]:
    # need more space for buffers that are created during quantization
    max_memory = {key: val * 0.90 for key, val in max_memory.items()}
    return max_memory
</syntaxhighlight>

CPU offload validation from `quantizer_bnb_8bit.py:68-79`:

<syntaxhighlight lang="python">
device_map = kwargs.get("device_map")
if not self.quantization_config.llm_int8_enable_fp32_cpu_offload and isinstance(device_map, dict):
    values = set(device_map.values())
    if values != {"cpu"} and ("cpu" in values or "disk" in values):
        raise ValueError(
            "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM "
            "to fit the quantized model. If you want to dispatch the model on the CPU or the disk "
            "while keeping these modules in 32-bit, you need to set "
            "`llm_int8_enable_fp32_cpu_offload=True`..."
        )
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: bitsandbytes not found` || BnB not installed || `pip install bitsandbytes>=0.43.0`
|-
|| `CUDA extension not compiled` || BnB compiled for wrong CUDA version || Reinstall BnB matching your CUDA version
|-
|| `ValueError: CPU offload not enabled` || Trying to offload to CPU without flag || Set `llm_int8_enable_fp32_cpu_offload=True`
|-
|| `RuntimeError: expected scalar type Half` || Dtype mismatch in quantized layers || Ensure `bnb_4bit_compute_dtype=torch.float16`
|}

== Compatibility Notes ==

* **Linux**: Full support; recommended platform
* **Windows**: Limited support; may require manual compilation
* **macOS**: Not supported (no CUDA)
* **AMD GPUs (ROCm)**: Experimental support in recent bitsandbytes versions
* **Quantized Training**: 8-bit and 4-bit models support training with PEFT adapters

== Related Pages ==

* [[requires_env::Implementation:huggingface_transformers_BitsAndBytesConfig]]
* [[requires_env::Implementation:huggingface_transformers_get_hf_quantizer]]
* [[requires_env::Implementation:huggingface_transformers_quantizer_preprocess_model]]
* [[requires_env::Implementation:huggingface_transformers_quantizer_postprocess_model]]
