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

**Purpose:** Generates CMakeUserPresets.json configuration file for streamlined vLLM development builds.

**Mechanism:** Detects system configuration (nvcc path, Python executable, CPU cores), calculates optimal build parallelism (NVCC_THREADS and cmake jobs based on CPU count), and generates JSON presets with compiler caching (sccache/ccache) and Ninja generator configuration. Creates "release" configure and build presets with detected paths.

**Significance:** Developer convenience tool that automates CMake configuration discovery. Eliminates manual CMake flag management and optimizes build performance by automatically setting parallelism parameters. Particularly valuable for new contributors who may not know optimal build settings.
