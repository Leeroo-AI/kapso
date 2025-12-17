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

**Purpose:** Environment diagnostic collection

**Mechanism:** Comprehensive system information gathering tool that collects details about the runtime environment: OS info, Python version, pip packages, conda packages, compiler versions (GCC, Clang, CMake), CUDA/ROCm configuration, GPU details, PyTorch installation, and vLLM-specific settings. Includes 30+ specialized functions for querying different aspects of the system. Can be run as a standalone script via __main__ to generate bug reports.

**Significance:** Essential debugging and support tool. When users report issues, this script generates standardized environment reports containing all relevant configuration details. Helps maintainers reproduce bugs and diagnose platform-specific problems. Critical for managing the complexity of supporting multiple hardware platforms and software configurations.
