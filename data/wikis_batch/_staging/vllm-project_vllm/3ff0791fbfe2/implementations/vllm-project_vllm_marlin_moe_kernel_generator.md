---
knowledge_sources:
  - vllm-project/vllm repository
  - File: csrc/moe/marlin_moe_wna16/generate_kernels.py
domains:
  - Code Generation
  - CUDA Kernels
  - Quantization
  - Mixture of Experts
  - Build System
last_updated: 2025-12-17
---

# Marlin MoE Kernel Generator

## Overview

Build-time code generator that creates CUDA kernel instantiations for Marlin MoE (Mixture of Experts) quantization with 16-bit activations across multiple GPU architectures and quantization schemes.

## Description

The `generate_kernels.py` script is a Python-based code generator that runs during vLLM's build process to create optimized CUDA kernel files for Marlin MoE inference. It uses Jinja2 templates to generate explicit template instantiations, enabling compile-time specialization rather than runtime dispatch.

**Key Features:**
- **Architecture detection**: Parses CUDA compute capabilities from command-line args
- **Multi-architecture support**: Generates SM75, SM80, SM89, and SM120 specific kernels
- **Quantization schemes**: AWQ-INT4, GPTQ-INT4, AWQ-INT8, FP8, NVFP4, MXFP4
- **Mixed precision**: Supports INT8/FP8 activations with INT4 weights
- **Configuration space**: Multiple thread configurations, block sizes, group sizes
- **Kernel selector**: Generates runtime selector for choosing optimal kernel

**Supported Quantization Combinations:**
- Standard: AWQ-INT4, GPTQ-INT4, AWQ-INT8, FP8
- Advanced: NVFP4 (4-bit float), MXFP4 (Microscaling FP4)
- Mixed: INT8 activations + INT4 weights, FP8 activations + INT4 weights

**Thread Configurations:**
- (128, 128, 256): Small batch, balanced
- (64, 256, 256): Large batch, more parallelism
- (64, 128, 128): Medium batch, conservative

**Generated Files:**
- `sm75_kernel_*.cu`: Turing (SM75) kernels with 2 stages
- `sm80_kernel_*.cu`: Ampere (SM80+) kernels with 4 stages
- `sm89_kernel_*.cu`: Ada/Hopper FP8 (SM89, SM120) kernels
- `kernel_selector.h`: Runtime kernel selection logic

## Usage

### Build System Integration

The script is invoked during CMake build:

```cmake
# CMakeLists.txt
execute_process(
    COMMAND ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe/marlin_moe_wna16/generate_kernels.py
        ${CMAKE_CUDA_ARCHITECTURES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
```

### Manual Invocation

```bash
# Generate kernels for specific architectures
python csrc/moe/marlin_moe_wna16/generate_kernels.py "8.0,8.9,9.0"

# Ampere only (SM80-86)
python csrc/moe/marlin_moe_wna16/generate_kernels.py "8.0,8.6"

# Hopper with FP8 support (SM89, SM90)
python csrc/moe/marlin_moe_wna16/generate_kernels.py "8.9,9.0"

# Legacy Turing (SM75)
python csrc/moe/marlin_moe_wna16/generate_kernels.py "7.5,8.0"
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/csrc/moe/marlin_moe_wna16/generate_kernels.py`

**Main Functions:**

```python
def remove_old_kernels() -> None:
    """Remove previously generated kernel files"""

def generate_new_kernels() -> None:
    """Generate new kernel instantiations and selector"""
```

**Architecture Detection:**

```python
ARCHS = []
SUPPORT_FP8 = False
SUPPORT_SM75 = False
SUPPORT_SM80 = False

for arch in sys.argv[1].split(","):
    arch = arch[: arch.index(".") + 2].replace(".", "")
    arch = int(arch)

    if arch in [89, 120]:  # Ada, Hopper
        SUPPORT_FP8 = True
    if arch >= 80:  # Ampere+
        SUPPORT_SM80 = True
    if arch == 75:  # Turing
        SUPPORT_SM75 = True
```

**Template Rendering:**

```python
TEMPLATE = (
    "template __global__ void Marlin<"
    "{{a_type_id}}, {{b_type_id}}, {{c_type_id}}, {{s_type_id}}, "
    "{{threads}}, {{thread_m_blocks}}, {{thread_n_blocks}}, "
    "{{thread_k_blocks}}, {{m_block_size_8}}, {{stages}}, "
    "{{group_blocks}}, {{is_zp_float}}>"
    "( MARLIN_KERNEL_PARAMS );"
)

template_str = jinja2.Template(TEMPLATE).render(
    a_type_id=f"vllm::{a_type}.id()",
    b_type_id=f"vllm::{b_type}.id()",
    c_type_id=f"vllm::{c_type}.id()",
    s_type_id=f"vllm::{s_type}.id()",
    **config
)
```

**Import:**
```python
import jinja2
import itertools
import subprocess
```

## I/O Contract

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `sys.argv[1]` | `str` | Comma-separated CUDA architectures (e.g., "8.0,8.9") |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `sm{XX}_kernel_*.cu` | Files | Generated CUDA kernel instantiations |
| `kernel_selector.h` | File | Runtime kernel selection logic |

### Quantization Config Structure

```python
{
    "a_type": list[str],         # Activation types: ["kFloat16", "kBFloat16", "kS8", "kFE4M3fn"]
    "b_type": str,               # Weight type: "kU4", "kU4B8", "kU8B128", "kFE4M3fn", "kFE2M1f"
    "c_type": list[str],         # Output types: ["kFloat16", "kBFloat16"]
    "s_type": str,               # Scale type: matches c_type or "kFE8M0fnu"
    "thread_configs": list,      # [(thread_k, thread_n, threads), ...]
    "thread_m_blocks": list,     # [0.5, 1, 2, 3, 4]
    "group_blocks": list,        # [-1, 0, 2, 4, 8]
}
```

### Generated Kernel Signature

```cpp
template <
    int a_type_id,           // Activation type ID
    int b_type_id,           // Weight type ID
    int c_type_id,           // Output type ID
    int s_type_id,           // Scale type ID
    int threads,             // Threads per block (128 or 256)
    int thread_m_blocks,     // M-dimension blocks per thread
    int thread_n_blocks,     // N-dimension blocks per thread
    int thread_k_blocks,     // K-dimension blocks per thread
    bool m_block_size_8,     // Use 8-element M blocks
    int stages,              // Pipeline stages (2 or 4)
    int group_blocks,        // Quantization group blocks (-1, 0, 2, 4, 8)
    bool is_zp_float         // Zero-point is float
>
__global__ void Marlin(MARLIN_KERNEL_PARAMS);
```

### Kernel Selector Logic

```cpp
// kernel_selector.h
if (a_type == vllm::kFloat16 && b_type == vllm::kU4 &&
    c_type == vllm::kFloat16 && s_type == vllm::kFloat16 &&
    threads == 128 && thread_m_blocks == 1 &&
    thread_n_blocks == 8 && thread_k_blocks == 4 &&
    m_block_size_8 == false && stages == 4 &&
    group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<...>;
else if (...)
  kernel = Marlin<...>;
...
```

## Usage Examples

### Building with Specific Architectures

```bash
# Build for A100 (SM80) and H100 (SM90)
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;90"
make

# Kernels generated for SM80 and SM89 (FP8 enabled)
ls ../csrc/moe/marlin_moe_wna16/sm*.cu
# sm80_kernel_float16_u4_float16.cu
# sm89_kernel_fe4m3fn_u4_float16.cu
# ...
```

### Examining Generated Kernels

```bash
# Generate kernels
python csrc/moe/marlin_moe_wna16/generate_kernels.py "8.0,8.9"

# Check generated files
ls csrc/moe/marlin_moe_wna16/

# View SM80 FP16 INT4 kernel
cat csrc/moe/marlin_moe_wna16/sm80_kernel_float16_u4_float16.cu

# View kernel selector
cat csrc/moe/marlin_moe_wna16/kernel_selector.h
```

### Understanding Kernel Configurations

```python
# Example configuration for AWQ-INT4
{
    "b_type": "kU4",                        # 4-bit unsigned weights
    "thread_configs": [
        (128, 128, 256),                    # Small batch
        (64, 256, 256),                     # Large batch
    ],
    "thread_m_blocks": [0.5, 1, 2, 3, 4],  # Batch size tuning
    "group_blocks": [-1, 2, 4, 8],         # Group quantization
}

# Generates kernels like:
# - Marlin<kFloat16, kU4, kFloat16, kFloat16, 256, 1, 8, 8, false, 4, -1, false>
# - Marlin<kFloat16, kU4, kFloat16, kFloat16, 256, 2, 16, 4, false, 4, 2, false>
# - ...
```

### Custom Quantization Scheme

```python
# Add to QUANT_CONFIGS in generate_kernels.py
{
    "a_type": ["kFloat16"],
    "b_type": "kU4",  # 4-bit weights
    "c_type": ["kBFloat16"],  # BF16 output
    "thread_configs": [(128, 128, 256)],
    "thread_m_blocks": [1, 2],
    "group_blocks": [4],  # 4-block groups
}

# Regenerate
python csrc/moe/marlin_moe_wna16/generate_kernels.py "8.0"

# New file created:
# sm80_kernel_float16_u4_bfloat16.cu
```

### Architecture-Specific Features

```python
# FP8 support only on SM89 and SM120
if arch in [89, 120]:
    SUPPORT_FP8 = True

# Generates FP8 kernels:
{
    "a_type": ["kFE4M3fn"],  # FP8 E4M3
    "b_type": "kFE4M3fn",
    "c_type": ["kBFloat16"],
    "thread_configs": THREAD_CONFIGS,
    "thread_m_blocks": [1, 2, 3, 4],
    "group_blocks": [-1, 8],
}

# Only generated when --cuda-arch includes 8.9 or 12.0
```

### Debugging Kernel Selection

```cpp
// In vLLM code, add logging to see which kernel is selected
void run_marlin_moe_kernel(
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& c,
    // ... other params
) {
    // Kernel selection happens here
    decltype(&Marlin<...>) kernel = nullptr;

    // Generated selector logic from kernel_selector.h
    #include "kernel_selector.h"

    if (kernel == nullptr) {
        std::cerr << "No matching kernel for config: "
                  << "a=" << a.dtype() << ", b=" << b.dtype()
                  << ", threads=" << threads
                  << ", stages=" << stages << std::endl;
        throw std::runtime_error("Kernel not found");
    }

    // Launch kernel
    kernel<<<grid, block>>>(/* ... */);
}
```

### Performance Tuning

```python
# For small batches (M=1-16), prefer 128 threads
if m_blocks <= 1 and (thread_k, thread_n) != (128, 128):
    continue

# For large batches (M>16), prefer 256 threads
if m_blocks > 1 and (thread_k, thread_n) != (64, 256):
    continue

# This logic in generate_new_kernels() limits kernel combinations
# to avoid explosion of .cu files
```

### Cleaning Build Artifacts

```bash
# Remove old generated kernels
python -c "
from csrc.moe.marlin_moe_wna16.generate_kernels import remove_old_kernels
remove_old_kernels()
"

# Or manually
rm -f csrc/moe/marlin_moe_wna16/sm*_kernel_*.cu
rm -f csrc/moe/marlin_moe_wna16/kernel_selector.h

# Regenerate
python csrc/moe/marlin_moe_wna16/generate_kernels.py "8.0,8.9"
```

### Analyzing Generated Code Size

```bash
# Generate kernels
python csrc/moe/marlin_moe_wna16/generate_kernels.py "8.0,8.9,9.0"

# Count instantiations
echo "Total kernel files:"
ls csrc/moe/marlin_moe_wna16/sm*.cu | wc -l

# Count lines per file
for f in csrc/moe/marlin_moe_wna16/sm*.cu; do
    lines=$(wc -l < "$f")
    echo "$(basename $f): $lines lines"
done

# Analyze kernel selector complexity
echo "Kernel selector branches:"
grep -c "else if" csrc/moe/marlin_moe_wna16/kernel_selector.h
```

### Understanding Pipeline Stages

```python
# SM75 (Turing): 2 stages (limited shared memory)
if SUPPORT_SM75:
    config_sm75 = config.copy()
    config_sm75["stages"] = 2
    sm_75_result_dict[(a_type, b_type, c_type)].append(config_sm75)

# SM80+ (Ampere+): 4 stages (more shared memory, better pipelining)
if SUPPORT_SM80:
    config["stages"] = 4
    result_dict[(a_type, b_type, c_type)].append(config)

# More stages = better latency hiding but more shared memory usage
```

## Related Pages

- [[vllm-project_vllm_marlin_moe]] - Marlin MoE inference implementation
- [[vllm-project_vllm_moe_quantization]] - MoE quantization schemes
- [[vllm-project_vllm_awq_quantization]] - AWQ quantization method
- [[vllm-project_vllm_gptq_quantization]] - GPTQ quantization method
- [[vllm-project_vllm_fp8_quantization]] - FP8 quantization support
- [[vllm-project_vllm_cuda_kernel_generation]] - CUDA kernel code generation
- [[vllm-project_vllm_build_system]] - vLLM build system
