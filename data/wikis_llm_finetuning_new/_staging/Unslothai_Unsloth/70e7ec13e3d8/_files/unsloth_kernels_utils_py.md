# File: `unsloth/kernels/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1034 |
| Functions | `is_cdna`, `calculate_settings`, `QUANT_STATE`, `get_lora_parameters`, `get_lora_parameters_bias`, `fast_linear_forward`, `matmul_lora` |
| Imports | bitsandbytes, ctypes, device_type, fp8, functools, importlib, torch, triton, typing, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central utility module providing low-level GPU operations, quantization support, LoRA parameter extraction, and hardware abstraction for CUDA/HIP/XPU devices.

**Mechanism:** Key components: (1) calculate_settings() computes optimal Triton block sizes and warp counts based on input dimensions. (2) fast_dequantize() handles bitsandbytes 4-bit NF4 dequantization via CUDA streams, supports both old list and new class quant_state formats, includes FP8 dequant via weight_dequant(). Uses global buffers for inference optimization. (3) fast_gemv() performs optimized matrix-vector products for seq_len=1 using bitsandbytes' cgemm_4bit_inference. (4) get_lora_parameters() extracts (W, W_quant, A, B, scale) from peft LoRA layers, handles QAT fake quantization. (5) matmul_lora() combines quantized base weight matmul with LoRA computation: out = X@W.T + s*(X@A.T)@B.T. (6) fast_linear_forward() wraps complete linear+LoRA forward. Hardware abstraction for CUDA/XPU streams, torch.amp decorators versioned for torch < 2.4. Imports triton_tanh from correct location based on triton version.

**Significance:** Foundation utility layer for all kernel operations. Abstracts device-specific code, provides efficient dequantization critical for 4-bit training, and centralizes LoRA parameter handling used by fast_lora.py.
