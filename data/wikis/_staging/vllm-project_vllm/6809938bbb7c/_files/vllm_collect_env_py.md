# File: `vllm/collect_env.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 857 |
| Functions | `run`, `run_and_read_all`, `run_and_parse_first_match`, `run_and_return_first_line`, `get_conda_packages`, `get_gcc_version`, `get_clang_version`, `get_cmake_version`, `... +28 more` |
| Imports | collections, datetime, locale, os, regex, subprocess, sys, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** System environment diagnostics and debugging information collector.

**Mechanism:** Comprehensive utility for gathering system, hardware, and software information to aid in bug reporting and troubleshooting. Collects data about: (1) OS and Python version, (2) GPU information (CUDA, ROCm, drivers), (3) Compiler versions (GCC, Clang, NVCC), (4) Build tools (CMake, ninja), (5) Python packages (PyTorch, vLLM, transformers), (6) Environment variables, (7) Hardware specs (CPU, RAM, GPU memory). Functions like `get_nvidia_smi()`, `get_rocm_version()`, `get_torch_version()` query specific components. Can be run as a standalone script (`python -m vllm.collect_env`) to generate diagnostic reports.

**Significance:** Essential debugging and support tool. When users report issues, this script collects all relevant environment information in one place, making it easier to reproduce and diagnose problems. Particularly important for a library that depends on complex hardware/software stacks (CUDA, ROCm, specific PyTorch versions, etc.). Standardizes bug reports and accelerates issue resolution.
