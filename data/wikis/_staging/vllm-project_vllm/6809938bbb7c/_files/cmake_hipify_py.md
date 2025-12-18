# File: `cmake/hipify.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 80 |
| Imports | argparse, os, shutil, torch |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** CUDA to ROCm/HIP source code translator

**Mechanism:** Command-line wrapper around PyTorch's hipify preprocessor. Takes CUDA source files (.cu, .cuh) and converts them to HIP equivalents for AMD GPUs. Copies entire project directory to output directory, applies hipify transformations only to specified source files (not all files), limits include scope to project directory, and prints list of hipified source paths for CMake integration. Uses `hipify_extra_files_only=True` to avoid transforming the entire codebase.

**Significance:** Essential for building vLLM on AMD ROCm platforms. Enables automatic translation of CUDA kernels to HIP, allowing vLLM's extensive CUDA codebase to run on AMD GPUs without manual porting. Part of vLLM's multi-backend strategy supporting both NVIDIA and AMD hardware. The selective hipification approach maintains build performance by only converting specified files.
