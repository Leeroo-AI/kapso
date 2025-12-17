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

**Purpose:** Generates optimized CUTLASS-based kernels for Machete quantization with automatic tile size selection heuristics.

**Mechanism:** Creates dispatch logic and kernel implementations for GPTQ and AWQ quantization schemes. Uses sophisticated tile heuristics based on M/K/N dimensions to select optimal tile shapes (e.g., 128x128 for M>256, 128x64 for M>128 with constraints). Splits kernel implementations across multiple files (8 by default) to manage compilation parallelism. Generates three types of files: dispatch, prepack, and implementation parts.

**Significance:** Most sophisticated quantization kernel generator in vLLM. The dimension-based heuristics (tuned for H100) enable automatic performance optimization without manual kernel selection. Supports advanced features like group scales, zero points, channel scales, and token scales for flexible quantization strategies.
