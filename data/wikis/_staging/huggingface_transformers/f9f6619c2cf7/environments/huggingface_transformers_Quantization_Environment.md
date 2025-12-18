# Environment: huggingface_transformers_Quantization_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Quantization Documentation|https://huggingface.co/docs/transformers/quantization]]
|-
! Domains
| [[domain::Quantization]], [[domain::Optimization]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
GPU-accelerated environment for loading quantized models (4-bit, 8-bit, FP8) with bitsandbytes, GPTQ, AWQ, or other quantization backends.

=== Description ===
This environment enables loading and running inference with quantized models that use reduced precision weights. It supports multiple quantization methods: bitsandbytes (INT8/INT4), GPTQ, AWQ, EETQ, HQQ, and FP8 formats. Quantization significantly reduces memory usage (2-4x) at minimal accuracy cost, enabling larger models on consumer GPUs.

=== Usage ===
Use this environment when loading models with `BitsAndBytesConfig`, `GPTQConfig`, `AwqConfig`, or other quantization configs. Required for running 7B+ parameter models on consumer GPUs (16GB VRAM or less).

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux || Windows has limited support
|-
| Python || Python >= 3.10 || Required by transformers
|-
| Hardware || NVIDIA GPU (Ampere+) || CUDA compute capability >= 7.0
|-
| GPU Memory || 8GB+ || Depends on model size and quantization
|-
| CUDA || CUDA >= 11.8 || For bitsandbytes
|}

== Dependencies ==

=== System Packages ===
* `cuda-toolkit` >= 11.8 - Required for bitsandbytes

=== Python Packages ===
* `transformers` (this package)
* `torch` >= 2.2
* `accelerate` >= 1.1.0 - **Required** for quantization

=== Quantization Backends (choose one or more) ===
* `bitsandbytes` - For INT8/INT4 (LLM.int8(), QLoRA-style)
* `auto-gptq` - For GPTQ quantized models
* `autoawq` - For AWQ quantized models
* `optimum` - For EETQ, HQQ, and other methods
* `torchao` - For FP8 and other PyTorch-native quantization

== Credentials ==
* `HF_TOKEN`: For gated quantized models

== Quick Install ==

<syntaxhighlight lang="bash">
# With bitsandbytes (4-bit/8-bit)
pip install transformers torch accelerate bitsandbytes

# With GPTQ support
pip install transformers torch accelerate auto-gptq optimum

# With AWQ support
pip install transformers torch accelerate autoawq

# With HQQ support
pip install transformers torch accelerate hqq
</syntaxhighlight>

== Code Evidence ==

Bitsandbytes validation from `quantizer_bnb_8bit.py:L54-66`:

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

Device map auto-detection from `quantizer_bnb_8bit.py:L86-103`:

<syntaxhighlight lang="python">
def update_device_map(self, device_map):
    if device_map is None:
        if torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}
        elif is_torch_npu_available():
            device_map = {"": f"npu:{torch.npu.current_device()}"}
        elif is_torch_hpu_available():
            device_map = {"": f"hpu:{torch.hpu.current_device()}"}
        elif is_torch_xpu_available():
            device_map = {"": torch.xpu.current_device()}
        else:
            device_map = {"": "cpu"}
    return device_map
</syntaxhighlight>

Supported quantization methods from `quantizers/auto.py:L65-86`:

<syntaxhighlight lang="python">
AUTO_QUANTIZER_MAPPING = {
    "awq": AwqQuantizer,
    "bitsandbytes_4bit": Bnb4BitHfQuantizer,
    "bitsandbytes_8bit": Bnb8BitHfQuantizer,
    "gptq": GptqHfQuantizer,
    "aqlm": AqlmHfQuantizer,
    "quanto": QuantoHfQuantizer,
    "eetq": EetqHfQuantizer,
    "hqq": HqqHfQuantizer,
    "compressed-tensors": CompressedTensorsHfQuantizer,
    "fbgemm_fp8": FbgemmFp8HfQuantizer,
    "torchao": TorchAoHfQuantizer,
    "fp8": FineGrainedFP8HfQuantizer,
    # ... more methods
}
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: bitsandbytes is required` || bitsandbytes not installed || `pip install bitsandbytes`
|-
|| `RuntimeError: CUDA is required for bitsandbytes` || No CUDA GPU available || Quantization requires GPU
|-
|| `ImportError: accelerate is required` || Accelerate not installed || `pip install accelerate>=1.1.0`
|-
|| `ValueError: Unknown quantization type` || Unsupported quant method || Check `AUTO_QUANTIZER_MAPPING`
|-
|| `CUDA extension not installed` || bitsandbytes compilation issue || `pip install -U bitsandbytes` or build from source
|-
|| `OutOfMemoryError` even with quantization || Model still too large || Use 4-bit instead of 8-bit, or smaller model
|}

== Compatibility Notes ==

* **NVIDIA GPUs:** Best support; required for most quantization methods
* **AMD GPUs (ROCm):** Limited bitsandbytes support (>= 0.48.3)
* **Intel XPU:** Some methods via optimum-intel
* **CPU:** Most quantization methods require GPU for inference
* **Windows:** bitsandbytes has limited Windows support
* **Compute Dtype:** Use `bnb_4bit_compute_dtype=torch.bfloat16` for Ampere+ GPUs
* **Double Quant:** `bnb_4bit_use_double_quant=True` reduces memory further

== Usage Example ==

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto",
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Implementation:huggingface_transformers_BitsAndBytesConfig_setup]]
* [[requires_env::Implementation:huggingface_transformers_AutoHfQuantizer_dispatch]]
* [[requires_env::Implementation:huggingface_transformers_Quantizer_validate_environment]]
* [[requires_env::Implementation:huggingface_transformers_Quantizer_preprocess]]
* [[requires_env::Implementation:huggingface_transformers_Quantizer_convert_weights]]
* [[requires_env::Implementation:huggingface_transformers_Skip_modules_handling]]
* [[requires_env::Implementation:huggingface_transformers_Quantizer_postprocess]]
