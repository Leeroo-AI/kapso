# File: `csrc/cutlass_extensions/vllm_cutlass_library_extension.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 76 |
| Classes | `VLLMDataType`, `MixedInputKernelScheduleType` |
| Imports | cutlass_library, enum |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extends NVIDIA CUTLASS library with vLLM-specific custom data types and kernel schedules.

**Mechanism:** Defines VLLMDataType enum with custom types (u4b8, u8b128) for GPTQ-style quantization, and MixedInputKernelScheduleType for TMA warp-specialized kernel schedules. Provides mapping dictionaries (VLLMDataTypeNames, VLLMDataTypeTag, VLLMDataTypeSize) to translate between vLLM types, CUTLASS types, and PyTorch scalar types.

**Significance:** Core infrastructure for quantization kernel code generation. These type definitions are consumed by machete and marlin kernel generators to produce specialized CUDA kernels for different quantization formats (INT4, INT8, FP8, GPTQ).
