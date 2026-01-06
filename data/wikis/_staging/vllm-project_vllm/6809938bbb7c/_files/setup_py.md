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

**Purpose:** Comprehensive build system for vLLM supporting multiple platforms (CUDA, ROCm, CPU, TPU, XPU) with CMake-based C++/CUDA compilation.

**Mechanism:** Key components:
- **CMakeExtension/cmake_build_ext:** Custom setuptools extension using CMake for building C++/CUDA code with sccache/ccache support, parallel compilation, and Ninja build system
- **precompiled_wheel_utils:** Downloads and extracts precompiled binaries from nightly wheel repository to speed up development builds
- **Platform detection:** _is_cuda(), _is_hip(), _is_cpu(), etc. for conditional compilation
- **Version management:** get_vllm_version() appends platform suffixes (cu124, rocm60, cpu, tpu)
- **Requirements handling:** get_requirements() loads platform-specific dependencies

**Significance:** The central build orchestrator. Enables vLLM to be built from source on any supported platform, with precompiled binary fallback for rapid development iteration. Critical for the CI/CD pipeline and end-user installations.
