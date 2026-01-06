# File: `tools/generate_cmake_presets.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 180 |
| Functions | `get_python_executable`, `get_cpu_cores`, `generate_presets` |
| Imports | argparse, json, multiprocessing, os, shutil, sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** CMake configuration generator for vLLM builds

**Mechanism:** Auto-detects build environment and generates `CMakeUserPresets.json` with optimized settings. Detects: (1) NVCC path from PyTorch's CUDA_HOME or PATH, (2) Python executable from current environment, (3) CPU cores for parallelism tuning (NVCC_THREADS and CMake jobs), (4) compiler cache tools (sccache/ccache), (5) Ninja generator availability. Creates preset with Release configuration, proper compiler paths, and parallelism flags. Prompts for user confirmation before overwriting existing files.

**Significance:** Simplifies vLLM's complex CUDA build process by automating environment detection and creating optimized CMake configurations. Reduces build time by properly configuring parallel compilation, compiler caching, and Ninja generator. Essential for developers building vLLM from source, especially on varied development environments.
