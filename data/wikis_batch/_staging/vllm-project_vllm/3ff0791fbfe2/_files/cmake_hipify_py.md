# File: `cmake/hipify.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 80 |
| Imports | argparse, os, shutil, torch |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Converts CUDA source code to ROCm/HIP equivalents for AMD GPU compatibility.

**Mechanism:** Wraps PyTorch's hipify_python tool to translate CUDA code. Copies entire project directory to output location, then runs hipify on specified source files with project-scoped includes. Prints hipified file paths for CMake build integration.

**Significance:** Critical for ROCm backend support. Enables vLLM to support AMD GPUs without maintaining separate codebases. The project-directory-scoped includes ensure header dependencies are correctly translated alongside source files.
