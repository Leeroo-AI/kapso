{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Diagnostics]], [[domain::System_Information]], [[domain::Debugging]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Comprehensive system environment information collector for debugging and issue reporting in vLLM.

=== Description ===
The collect_env.py module is a 857-line diagnostic utility that gathers detailed information about the system environment, hardware configuration, software versions, and installed packages. This module is essential for debugging vLLM issues, as it captures all relevant environmental factors that might affect inference behavior.

The module collects information across multiple categories: (1) Operating system details (Linux distribution, macOS version, Windows version); (2) Hardware information including CPU details (architecture, cores, caches, NUMA topology), GPU models and driver versions (NVIDIA/AMD), and GPU topology for multi-GPU systems; (3) Software versions including Python, PyTorch, CUDA/ROCm toolkits, cuDNN, and compiler versions (GCC, Clang, CMake); (4) Python packages especially those relevant to ML/AI (torch, numpy, transformers, triton, NCCL, etc.); (5) Environment variables related to PyTorch, CUDA, NCCL, and vLLM configuration; (6) vLLM-specific information including version, build flags, and CUDA architecture list.

The output is formatted as a comprehensive multi-section report that can be easily shared when reporting issues. The module is based on PyTorch's collect_env.py but extended with vLLM-specific information gathering. It handles various edge cases like missing tools, permission errors, and different OS platforms.

=== Usage ===
Run as a standalone script or import to programmatically collect environment information for debugging.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/collect_env.py vllm/collect_env.py]
* '''Lines:''' 1-857

=== Signature ===
<syntaxhighlight lang="python">
# Main data structure
SystemEnv = namedtuple("SystemEnv", [
    "torch_version", "is_debug_build", "cuda_compiled_version",
    "gcc_version", "clang_version", "cmake_version", "os",
    "libc_version", "python_version", "python_platform",
    "is_cuda_available", "cuda_runtime_version", "cuda_module_loading",
    "nvidia_driver_version", "nvidia_gpu_models", "cudnn_version",
    "pip_version", "pip_packages", "conda_packages",
    "hip_compiled_version", "hip_runtime_version", "miopen_runtime_version",
    "caching_allocator_config", "is_xnnpack_available", "cpu_info",
    "rocm_version", "vllm_version", "vllm_build_flags", "gpu_topo",
    "env_vars",
])

# Main collection functions
def get_env_info() -> SystemEnv
def get_pretty_env_info() -> str
def pretty_str(envinfo: SystemEnv) -> str

# Helper functions
def run(command) -> tuple[int, str, str]
def get_gpu_info(run_lambda) -> str | None
def get_nvidia_driver_version(run_lambda) -> str | None
def get_cuda_version(run_lambda) -> str | None
def get_rocm_version(run_lambda) -> str | None
def get_vllm_version() -> str
def get_gpu_topo(run_lambda) -> str | None
def get_cpu_info(run_lambda) -> str
def get_pip_packages(run_lambda, patterns=None) -> tuple[str, str]
def get_conda_packages(run_lambda, patterns=None) -> str | None
def get_env_vars() -> str
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.collect_env import get_env_info, get_pretty_env_info

# Collect environment information
env_info = get_env_info()
print(f"vLLM version: {env_info.vllm_version}")
print(f"PyTorch version: {env_info.torch_version}")
print(f"CUDA version: {env_info.cuda_compiled_version}")

# Get formatted output
print(get_pretty_env_info())
</syntaxhighlight>

== I/O Contract ==

=== Key Exports ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| SystemEnv || NamedTuple || Data structure holding all environment information
|-
| get_env_info || Function || Collect all environment information
|-
| get_pretty_env_info || Function || Get formatted environment report
|-
| pretty_str || Function || Format SystemEnv into readable string
|-
| get_gpu_info || Function || Get GPU model information
|-
| get_nvidia_driver_version || Function || Get NVIDIA driver version
|-
| get_rocm_version || Function || Get ROCm version
|-
| get_vllm_version || Function || Get vLLM version with git info
|-
| get_gpu_topo || Function || Get GPU topology information
|-
| get_cpu_info || Function || Get detailed CPU information
|-
| get_pip_packages || Function || List relevant pip packages
|-
| get_conda_packages || Function || List relevant conda packages
|-
| get_env_vars || Function || Collect relevant environment variables
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Example 1: Run as standalone script for debugging
# In terminal:
# python -m vllm.collect_env

# Example 2: Programmatic collection
from vllm.collect_env import get_env_info, get_pretty_env_info

# Get structured environment data
env_info = get_env_info()

# Access specific fields
print(f"vLLM Version: {env_info.vllm_version}")
print(f"Python Version: {env_info.python_version}")
print(f"PyTorch Version: {env_info.torch_version}")
print(f"CUDA Available: {env_info.is_cuda_available}")
print(f"CUDA Runtime: {env_info.cuda_runtime_version}")
print(f"GPU Models: {env_info.nvidia_gpu_models}")
print(f"ROCm Version: {env_info.rocm_version}")

# Get full formatted report
full_report = get_pretty_env_info()
print(full_report)

# Save to file for issue reporting
with open("environment_info.txt", "w") as f:
    f.write(full_report)

# Example 3: Check specific conditions
if env_info.is_cuda_available == "True":
    print("CUDA is available")
    if env_info.cuda_runtime_version:
        print(f"CUDA Runtime: {env_info.cuda_runtime_version}")
    if env_info.cudnn_version:
        print(f"cuDNN: {env_info.cudnn_version}")

if env_info.rocm_version and env_info.rocm_version != "Could not collect":
    print(f"Running on ROCm {env_info.rocm_version}")

# Example 4: Check GPU topology for multi-GPU setups
if env_info.gpu_topo:
    print("GPU Topology:")
    print(env_info.gpu_topo)
    # This shows NVLink/PCIe connections between GPUs

# Example 5: Examine installed packages
if env_info.pip_packages:
    print("Relevant pip packages:")
    for line in env_info.pip_packages.split('\n'):
        if 'torch' in line.lower() or 'vllm' in line.lower():
            print(f"  {line}")

# Example 6: Check vLLM build configuration
if env_info.vllm_build_flags:
    print(f"vLLM Build Flags: {env_info.vllm_build_flags}")
    # Shows CUDA architectures and ROCm status
</syntaxhighlight>

== Related Pages ==
* [[uses::Module:vllm-project_vllm_Environment_Variables]]
* [[implements::Interface:vllm-project_vllm_Diagnostics]]
* [[related::Module:vllm-project_vllm_Logger]]
