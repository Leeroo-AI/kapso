# File: `csrc/quantization/machete/generate.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 694 |
| Classes | `ScheduleConfig`, `TypeConfig`, `PrepackTypeConfig`, `ImplConfig` |
| Functions | `generate_sch_sig`, `generate_terse_sch_sig`, `generate_type_signature`, `generate_type_option_name`, `is_power_of_two`, `to_cute_constant`, `unique_schedules`, `unsigned_type_with_bitwidth`, `... +3 more` |
| Imports | collections, copy, dataclasses, functools, itertools, jinja2, math, os, shutil, vllm_cutlass_library_extension |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Advanced code generator for Machete mixed-precision GEMM kernels using CUTLASS 3.x and TMA (Tensor Memory Accelerator).

**Mechanism:** Most sophisticated of the kernel generators. Defines TypeConfig for full type signatures (A, B, scales, zeros, channel/token scales), ScheduleConfig for tile/cluster shapes and scheduler types. Generates dispatch code with size-based heuristics (optimized for H100), implementation files split across multiple .cu files for parallel compilation, and prepack routines for weight layout transformation.

**Significance:** Machete represents vLLM's cutting-edge quantization technology supporting GPTQ, AWQ, and upcoming W4A8 (QQQ) formats with dynamic tile selection based on problem size. The heuristics are tuned for maximum H100 performance.
