# Environment Diagnostic Collection

**File:** `/tmp/praxium_repo_583nq7ea/vllm/collect_env.py`
**Type:** Diagnostic and Support Tool
**Lines of Code:** 857
**Language:** Python
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Overview

The collect_env module is a comprehensive system information gathering tool that collects details about the runtime environment for debugging and support. It captures OS info, Python version, pip/conda packages, compiler versions, CUDA/ROCm configuration, GPU details, PyTorch installation, and vLLM-specific settings.

This tool generates standardized environment reports that help maintainers reproduce bugs and diagnose platform-specific problems, making it essential for managing the complexity of supporting multiple hardware platforms and software configurations.

## Implementation

### Core Architecture

**SystemEnv NamedTuple:**
```python
SystemEnv = namedtuple(
    "SystemEnv",
    [
        "torch_version", "is_debug_build", "cuda_compiled_version",
        "gcc_version", "clang_version", "cmake_version",
        "os", "libc_version", "python_version", "python_platform",
        "is_cuda_available", "cuda_runtime_version", "cuda_module_loading",
        "nvidia_driver_version", "nvidia_gpu_models", "cudnn_version",
        "pip_version", "pip_packages", "conda_packages",
        "hip_compiled_version", "hip_runtime_version", "miopen_runtime_version",
        "caching_allocator_config", "is_xnnpack_available", "cpu_info",
        "rocm_version", "vllm_version", "vllm_build_flags", "gpu_topo", "env_vars",
    ],
)
```

### Key Functions

**1. GPU Information**
```python
def get_gpu_info(run_lambda):
    """Detect NVIDIA or AMD GPUs"""
    if get_platform() == "darwin" or (
        TORCH_AVAILABLE and hasattr(torch.version, "hip") and torch.version.hip
    ):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if torch.version.hip is not None:
                prop = torch.cuda.get_device_properties(0)
                if hasattr(prop, "gcnArchName"):
                    gcnArch = f" ({prop.gcnArchName})"
                else:
                    gcnArch = "NoGCNArchNameOnOldPyTorch"
            else:
                gcnArch = ""
            return torch.cuda.get_device_name(None) + gcnArch
        return None

    smi = get_nvidia_smi()
    uuid_regex = re.compile(r" \(UUID: .+?\)")
    rc, out, _ = run_lambda(smi + " -L")
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return re.sub(uuid_regex, "", out)

def get_gpu_topo(run_lambda):
    """Get GPU topology matrix"""
    output = None
    if get_platform() == "linux":
        output = run_and_read_all(run_lambda, "nvidia-smi topo -m")
        if output is None:
            output = run_and_read_all(run_lambda, "rocm-smi --showtopo")
    return output
```

**2. Compiler Versions**
```python
def get_gcc_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")

def get_clang_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "clang --version", r"clang version (.*)"
    )

def get_cmake_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")
```

**3. CUDA/ROCm Detection**
```python
def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "nvcc --version", r"release .+ V(.*)"
    )

def get_rocm_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "hipcc --version", r"HIP version: (\S+)"
    )

def get_nvidia_driver_version(run_lambda):
    if get_platform() == "darwin":
        cmd = "kextstat | grep -i cuda"
        return run_and_parse_first_match(
            run_lambda, cmd, r"com[.]nvidia[.]CUDA [(](.*?)[)]"
        )
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run_lambda, smi, r"Driver Version: (.*?) ")
```

**4. Python Package Detection**
```python
def get_pip_packages(run_lambda, patterns=None):
    if patterns is None:
        patterns = DEFAULT_PIP_PATTERNS

    def run_with_pip():
        try:
            import importlib.util
            pip_spec = importlib.util.find_spec("pip")
            pip_available = pip_spec is not None
        except ImportError:
            pip_available = False

        if pip_available:
            cmd = [sys.executable, "-mpip", "list", "--format=freeze"]
        elif is_uv_venv():
            cmd = ["uv", "pip", "list", "--format=freeze"]
        else:
            raise RuntimeError("Could not collect pip list output")

        out = run_and_read_all(run_lambda, cmd)
        return "\n".join(
            line for line in out.splitlines()
            if any(name in line for name in patterns)
        )

    pip_version = "pip3" if sys.version[0] == "3" else "pip"
    out = run_with_pip()
    return pip_version, out

DEFAULT_PIP_PATTERNS = {
    "torch", "numpy", "mypy", "flake8", "triton", "optree",
    "onnx", "nccl", "transformers", "zmq", "nvidia",
    "pynvml", "flashinfer-python",
}
```

**5. vLLM-Specific Information**
```python
def get_vllm_version():
    from vllm import __version__, __version_tuple__

    if __version__ == "dev":
        return "N/A (dev)"

    version_str = __version_tuple__[-1]
    if isinstance(version_str, str) and version_str.startswith("g"):
        # Dev build with git sha
        if "." in version_str:
            # Local changes
            git_sha = version_str.split(".")[0][1:]
            date = version_str.split(".")[-1][1:]
            return f"{__version__} (git sha: {git_sha}, date: {date})"
        else:
            # Clean dev build
            git_sha = version_str[1:]
            return f"{__version__} (git sha: {git_sha})"
    return __version__

def summarize_vllm_build_flags():
    return "CUDA Archs: {}; ROCm: {}".format(
        os.environ.get("TORCH_CUDA_ARCH_LIST", "Not Set"),
        "Enabled" if os.environ.get("ROCM_HOME") else "Disabled",
    )

def get_env_vars():
    """Collect relevant environment variables"""
    env_vars = ""
    secret_terms = ("secret", "token", "api", "access", "password")
    report_prefix = ("TORCH", "NCCL", "PYTORCH", "CUDA", "CUBLAS",
                     "CUDNN", "OMP_", "MKL_", "NVIDIA")

    for k, v in os.environ.items():
        if any(term in k.lower() for term in secret_terms):
            continue  # Skip sensitive variables
        if k in environment_variables or k.startswith(report_prefix):
            env_vars += f"{k}={v}\n"

    return env_vars
```

### Output Format

**Example Output:**
```
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.3 LTS (x86_64)
GCC version                  : gcc 11.4.0
Clang version                : clang version 14.0.0
CMake version                : cmake 3.27.7
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.4.0+cu121
Is debug build               : False
CUDA used to build PyTorch   : 12.1

==============================
      Python Environment
==============================
Python version               : 3.10.12 (64-bit runtime)
Python platform              : Linux-6.5.0-1025-aws-x86_64-with-glibc2.35

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : True
CUDA runtime version         : 12.1.105
GPU models and configuration :
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB
Nvidia driver version        : 535.129.03
cuDNN version                : 8.9.7

==============================
          CPU Info
==============================
Architecture:            x86_64
CPU(s):                  128
Model name:              Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz

==============================
Versions of relevant libraries
==============================
[pip3] torch==2.4.0+cu121
[pip3] transformers==4.45.0
[pip3] nvidia-nccl-cu12==2.20.5
[pip3] triton==3.0.0
[pip3] vllm==0.6.2+cu121

==============================
         vLLM Info
==============================
ROCM Version                 : N/A
vLLM Version                 : 0.6.2+cu121
vLLM Build Flags:
  CUDA Archs: 8.0;8.6;8.9;9.0; ROCm: Disabled
GPU Topology:
  	GPU0	GPU1
GPU0	 X 	NV12
GPU1	NV12	 X

==============================
     Environment Variables
==============================
CUDA_HOME=/usr/local/cuda-12.1
TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
NCCL_DEBUG=INFO
```

## Usage

**Command-Line:**
```bash
# Run as script
python -m vllm.collect_env

# Save to file
python -m vllm.collect_env > environment_report.txt
```

**Programmatic:**
```python
from vllm.collect_env import get_pretty_env_info

env_info = get_pretty_env_info()
print(env_info)
```

**In Bug Reports:**
```markdown
### Environment Information
```
[Paste output from python -m vllm.collect_env]
```
```

## Key Insights

**Design Philosophy:**
1. **Comprehensive Coverage:** Captures all aspects that could affect vLLM behavior
2. **Privacy-Aware:** Filters out sensitive environment variables (API keys, tokens)
3. **Anonymization:** Removes GPU UUIDs to protect hardware identifiers
4. **Robust Parsing:** Handles missing tools and platforms gracefully

**Why This Matters:**
- **Bug Reproduction:** Maintainers can recreate exact environment
- **Platform Issues:** Identifies CUDA/ROCm version mismatches
- **Performance Debugging:** GPU topology reveals communication bottlenecks
- **Support Efficiency:** Standardized format speeds up issue resolution

## Summary

The collect_env module is an essential diagnostic tool that captures comprehensive system information for debugging vLLM deployments. Its 30+ specialized functions query every aspect of the environment, from GPU topology to Python packages, generating standardized reports that enable efficient support and bug resolution.

Key capabilities:
- **30+ information collectors** for OS, Python, CUDA, ROCm, CPUs, GPUs
- **Privacy protection** via secret filtering and UUID anonymization
- **Robust platform detection** supporting CUDA, ROCm, CPU, TPU, XPU
- **Standardized output format** for efficient bug reports

This tool exemplifies infrastructure code that doesn't directly improve performance but dramatically improves maintainability and user support experience.
