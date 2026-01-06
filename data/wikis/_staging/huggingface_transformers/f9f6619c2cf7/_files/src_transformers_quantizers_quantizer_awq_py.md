# File: `src/transformers/quantizers/quantizer_awq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 95 |
| Classes | `AwqQuantizer` |
| Imports | base, importlib, packaging, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements 4-bit quantizer for AWQ (Activation-aware Weight Quantization) method, which protects important weights based on activation patterns.

**Mechanism:** The `AwqQuantizer` class extends `HfQuantizer` with `requires_calibration=True` for pre-quantized models only. `validate_environment` ensures gptqmodel and accelerate libraries are present. `update_dtype` forces float16 on CUDA/XPU since bfloat16 isn't supported by AWQ kernels. In preprocessing, `replace_with_awq_linear` converts linear layers to AWQ-specific implementations while respecting skip lists, then `replace_quantization_scales` applies model-type-specific scale handling. Post-loading calls `hf_gptqmodel_post_init` for finalization. Serialization blocked for Exllama backends. Training supported with gptqmodel>=5.0.0.

**Significance:** Enables AWQ quantization, a popular 4-bit method that preserves accuracy by protecting salient weights identified through activation analysis. Key component of the quantization ecosystem, offering efficient 4-bit inference with better quality than naive quantization.
