{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Build System]], [[domain::Package Management]], [[domain::Multi-Platform Compilation]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A sophisticated multi-platform build system for vLLM supporting CUDA, ROCm, CPU, TPU, and XPU with CMake-based C++ extension compilation.

=== Description ===
This comprehensive setup.py orchestrates vLLM's complex build process across multiple hardware platforms (NVIDIA CUDA, AMD ROCm, Intel XPU, Google TPU, and CPU). It implements a CMake-based build system with custom cmake_build_ext that handles CUDA/HIP kernel compilation, manages precompiled wheel downloads for faster development, detects and configures compilation caching (sccache/ccache), and sets platform-specific version suffixes. The build supports Ninja for parallel compilation, handles tensor core feature detection, manages Flash Attention dependencies, and implements freethreaded Python (PEP 703) support. It automatically determines optimal parallelism and NVCC thread counts for efficient compilation.

=== Usage ===
This is vLLM's primary installation entry point, used by pip install, for local development builds, and by CI/CD for package creation across all supported platforms.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/setup.py setup.py]

=== Signature ===
<syntaxhighlight lang="python">
class cmake_build_ext(build_ext):
    def configure(self, ext: CMakeExtension) -> None
    def build_extensions(self) -> None
    def compute_num_jobs(self) -> tuple[int, int | None]

class precompiled_build_ext(build_ext):
    # Skips compilation when using precompiled wheels

def get_vllm_version() -> str
def get_requirements() -> list[str]
def get_nvcc_cuda_version() -> Version
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Standard installation
pip install -e .

# Build with specific CUDA version
VLLM_MAIN_CUDA_VERSION=12.1 pip install -e .

# Use precompiled wheels
VLLM_USE_PRECOMPILED=1 pip install -e .

# Build for specific device
VLLM_TARGET_DEVICE=cuda pip install -e .
VLLM_TARGET_DEVICE=rocm pip install -e .
VLLM_TARGET_DEVICE=cpu pip install -e .
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| VLLM_TARGET_DEVICE || env || Target platform: cuda, rocm, cpu, tpu, xpu, empty
|-
| VLLM_USE_PRECOMPILED || env || Use precompiled binaries (1=yes)
|-
| CUDA_HOME || env || CUDA toolkit installation path
|-
| ROCM_HOME || env || ROCm installation path
|-
| MAX_JOBS || env || Maximum parallel compilation jobs
|-
| NVCC_THREADS || env || NVCC compilation threads (CUDA 11.2+)
|-
| CMAKE_BUILD_TYPE || env || CMake build type (Debug/RelWithDebInfo/Release)
|-
| CMAKE_ARGS || env || Additional CMake arguments
|-
| VLLM_VERSION_OVERRIDE || env || Override package version
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| vllm._C.abi3.so || Binary || Main CUDA/CPU extension module
|-
| vllm._moe_C.abi3.so || Binary || MoE-specific kernels
|-
| vllm._rocm_C.abi3.so || Binary || ROCm-specific kernels (ROCm only)
|-
| vllm.vllm_flash_attn._vllm_fa2_C.abi3.so || Binary || Flash Attention 2 kernels
|-
| vllm.vllm_flash_attn._vllm_fa3_C.abi3.so || Binary || Flash Attention 3 kernels (CUDA 12.3+)
|-
| vllm._flashmla_C.abi3.so || Binary || FlashMLA kernels (optional, Hopper+)
|-
| vllm.cumem_allocator.abi3.so || Binary || Custom CUDA memory allocator
|-
| vllm/_version.py || File || Generated version information
|}

== Platform Support ==

{| class="wikitable"
|-
! Platform !! Extensions Built !! Version Suffix !! Requirements
|-
| CUDA || _C, _moe_C, cumem_allocator, vllm_flash_attn || +cu{version} || CUDA 11.8+
|-
| ROCm || _C, _moe_C, _rocm_C, cumem_allocator || +rocm{version} || ROCm 5.7+
|-
| CPU || _C || +cpu || x86_64 or ARM64
|-
| TPU || None (Python-only) || +tpu || TPU VMs
|-
| XPU || _C || +xpu || Intel Data Center GPU
|-
| Empty || None || +empty || Package without compiled extensions
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Standard CUDA installation
pip install -e .

# Install with specific CUDA version (avoid version suffix)
VLLM_MAIN_CUDA_VERSION=12.1 pip install -e .

# ROCm installation
VLLM_TARGET_DEVICE=rocm pip install -e .

# CPU-only installation
VLLM_TARGET_DEVICE=cpu pip install -e .

# Use precompiled wheels for faster development
VLLM_USE_PRECOMPILED=1 pip install -e .

# Specify precompiled wheel location
VLLM_PRECOMPILED_WHEEL_LOCATION=/path/to/wheel.whl pip install -e .

# Custom build configuration
CMAKE_BUILD_TYPE=Debug \
MAX_JOBS=8 \
NVCC_THREADS=4 \
pip install -e .

# Enable verbose build output
VERBOSE=1 pip install -e .

# Build with custom CMake args
CMAKE_ARGS="-DVLLM_GPU_LANG=HIP" pip install -e .

# Parallel builds with Ninja (auto-detected)
# Uses sccache if available, else ccache
pip install -e .  # Automatically uses Ninja if installed
</syntaxhighlight>

== Build Optimizations ==

{| class="wikitable"
|-
! Feature !! Condition !! Benefit
|-
| sccache || sccache in PATH || Distributed compilation caching
|-
| ccache || ccache in PATH (no sccache) || Local compilation caching
|-
| Ninja || ninja in PATH || Faster parallel builds
|-
| NVCC threads || CUDA 11.2+, NVCC_THREADS set || Parallel NVCC compilation
|-
| Precompiled wheels || VLLM_USE_PRECOMPILED=1 || Skip compilation entirely
|}

== Related Pages ==
* [[Build:CMake_Configuration]]
* [[Build:CUDA_Compilation]]
* [[Build:Multi_Platform_Support]]
* [[Tool:CMake]]
* [[Tool:Ninja_Build]]
* [[Concept:Setuptools_Extensions]]
