# File: `setup.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 813 |
| Classes | `CMakeExtension`, `cmake_build_ext`, `precompiled_build_ext`, `precompiled_wheel_utils` |
| Functions | `load_module_from_path`, `is_sccache_available`, `is_ccache_available`, `is_ninja_available`, `is_freethreaded`, `get_rocm_version`, `get_nvcc_cuda_version`, `get_vllm_version`, `... +1 more` |
| Imports | ctypes, importlib, json, logging, os, packaging, pathlib, re, setuptools, setuptools_scm, ... +5 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main build configuration for vLLM package supporting multiple hardware backends (CUDA, ROCm, CPU, TPU, XPU) and precompiled wheels.

**Mechanism:** Implements custom CMakeExtension and cmake_build_ext classes to orchestrate CMake-based builds. Detects platform (CUDA/ROCm/CPU/etc.) and configures appropriate extensions (_C, _moe_C, flash_attn, etc.). Supports precompiled wheel workflow by downloading from wheels.vllm.ai and extracting compiled binaries. Handles compiler caching (sccache/ccache), parallel builds (MAX_JOBS/NVCC_THREADS), and version tagging with platform suffixes.

**Significance:** Central build orchestration script that makes vLLM buildable across diverse hardware platforms. The precompiled wheel support significantly reduces build times for users. Complex logic handles CUDA version detection, ROCm compatibility, and platform-specific dependencies, making vLLM accessible despite its complex C++/CUDA codebase.
