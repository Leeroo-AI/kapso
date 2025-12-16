# Environment: GGUF Export Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Model_Deployment]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

## Overview

Environment for exporting trained models to GGUF format using llama.cpp, enabling deployment with Ollama and llama.cpp inference.

### Description

This environment extends the base GPU environment to include llama.cpp compilation tools for GGUF export. It automatically clones, compiles, and uses llama.cpp for model conversion and quantization. The environment supports:
- CPU-only GGUF conversion (GPU not required for export)
- Multiple quantization methods (q4_k_m, q5_k_m, q8_0, etc.)
- Ollama Modelfile generation with proper chat templates

### Usage

Use this environment when:
- Exporting fine-tuned models to GGUF format
- Deploying models with Ollama or llama.cpp
- Creating quantized models for edge deployment

**Required for:**
- [[required_by::Implementation:unslothai_unsloth_save_to_gguf]]
- [[required_by::Workflow:unslothai_unsloth_Model_Export_GGUF]]

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+) || Windows/macOS supported with cmake
|-
| Hardware || CPU || GPU not required for GGUF conversion
|-
| Disk || 50GB+ free space || Required for temporary model files during conversion
|-
| RAM || 16GB+ || Recommended for larger model conversions
|}

## Dependencies

### System Packages

Required for llama.cpp compilation:

* `git` - For cloning llama.cpp repository
* `cmake` >= 3.14 - For cmake build system
* `make` - Alternative build system (legacy)
* `gcc/g++` >= 9.0 - C++ compiler with C++17 support

### Build Tools Detection

From `unsloth/save.py:870-906`:
```python
def install_llama_cpp_make_non_blocking():
    check = os.system("make clean -C llama.cpp")
    IS_CMAKE = False
    if check == 0:
        # Uses old MAKE
        n_jobs = max(int(psutil.cpu_count() * 1.5), 1)
        full_command = ["make", "all", "-j" + str(n_jobs), "-C", "llama.cpp"]
    else:
        # Uses new CMAKE
        n_jobs = max(int(psutil.cpu_count()), 1)
        check = os.system(
            "cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=ON"
        )
```

### Python Packages

* `unsloth_zoo` >= 2025.12.4 (provides llama.cpp utilities)

### llama.cpp Targets

From `unsloth/save.py:70-74`:
```python
LLAMA_CPP_TARGETS = [
    "llama-quantize",
    "llama-export-lora",
    "llama-cli",
]
```

## Credentials

* `HF_TOKEN`: Required for pushing to HuggingFace Hub
* Token must have write access for `push_to_hub=True`

## Supported Quantization Methods

From `unsloth/save.py:104-131`:

| Method | Description |
|--------|-------------|
| `q4_k_m` | **Recommended.** Uses Q6_K for half of attention.wv/feed_forward.w2, else Q4_K |
| `q5_k_m` | **Recommended.** Uses Q6_K for half of attention.wv/feed_forward.w2, else Q5_K |
| `q8_0` | Fast conversion. High resource use, but generally acceptable |
| `f16` | Float16 - Fastest conversion, retains 100% accuracy. Slow inference |
| `bf16` | Bfloat16 - Fastest conversion, retains 100% accuracy |
| `q2_k` | Uses Q4_K for attention.vw/feed_forward.w2, Q2_K for others |
| `q3_k_m` | Uses Q4_K for attention.wv/wo/feed_forward.w2, else Q3_K |

## Related Pages

### Required By

* [[required_by::Implementation:unslothai_unsloth_save_to_gguf]]
* [[required_by::Workflow:unslothai_unsloth_Model_Export_GGUF]]
