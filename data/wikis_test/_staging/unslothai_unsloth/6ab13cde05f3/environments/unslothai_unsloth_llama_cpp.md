# Environment: unslothai_unsloth_llama_cpp

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Doc|save.py|unsloth/save.py]]
* [[source::Doc|unsloth_zoo.llama_cpp|unsloth_zoo/llama_cpp.py]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Model_Export]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

## Overview

llama.cpp build environment for GGUF model export with quantization support, enabling local inference via llama.cpp and Ollama.

### Description

This environment provides the llama.cpp compilation toolchain needed for GGUF model export. Unsloth can automatically clone and build llama.cpp if not present, but this requires build tools. The environment supports multiple quantization methods including q4_k_m, q8_0, f16, and others.

Key components:
- **llama-quantize**: Quantizes merged models to various GGUF formats
- **llama-export-lora**: Exports LoRA adapters separately
- **llama-cli**: CLI for inference testing and validation
- **convert_hf_to_gguf.py**: Python script for HuggingFace to GGUF conversion

### Usage

Use this environment for:
- **GGUF Export**: `save_pretrained_gguf()` to convert models to GGUF format
- **Quantization**: Converting 16-bit merged models to 4-bit/8-bit GGUF
- **Ollama Deployment**: Generating Modelfiles for Ollama integration
- **Local Inference**: Testing models with llama.cpp CLI

Required when calling:
```python
model.save_pretrained_gguf(
    save_directory="output",
    quantization_method="q4_k_m"
)
```

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+), macOS || Windows requires WSL2
|-
| CPU || Multi-core recommended || Uses all available cores for compilation
|-
| RAM || 16GB+ || Quantization can be memory-intensive
|-
| Disk || 5GB+ free space || For llama.cpp build and intermediate files
|-
| Build Tools || CMake 3.14+, GCC 9+, Make || Auto-detected; CMAKE preferred for newer llama.cpp
|}

## Dependencies

### System Packages

* `git` - For cloning llama.cpp repository
* `cmake` >= 3.14 - Modern build system
* `make` - Legacy build system (fallback)
* `gcc`, `g++` >= 9 - C/C++ compilers
* `python3` >= 3.8 - For convert scripts

### Python Packages

* `psutil` - CPU core detection for parallel builds
* `requests` - For downloading dependencies
* `sentencepiece` - Tokenizer conversion

### Optional Packages

* `curl` development libraries - For `LLAMA_CURL=ON` (remote file support)

### Build Targets

From `save.py:70-74`:
```python
LLAMA_CPP_TARGETS = [
    "llama-quantize",
    "llama-export-lora",
    "llama-cli",
]
```

## Credentials

* `HF_TOKEN`: Required if pushing GGUF files to HuggingFace Hub

## Quantization Methods

From `save.py:104-131`:
```python
ALLOWED_QUANTS = {
    "not_quantized": "Fast conversion. Slow inference, big files.",
    "fast_quantized": "Fast conversion. OK inference, OK file size.",
    "quantized": "Slow conversion. Fast inference, small files.",
    "f32": "Retains 100% accuracy, but super slow and memory hungry.",
    "bf16": "Bfloat16 - Fastest conversion + retains 100% accuracy.",
    "f16": "Float16 - Fastest conversion + retains 100% accuracy.",
    "q8_0": "Fast conversion. High resource use, but acceptable.",
    "q4_k_m": "Recommended. Uses Q6_K for half of attention/feed_forward.",
    "q5_k_m": "Recommended. Uses Q6_K for half of attention/feed_forward.",
    # ... additional methods
}
```

## Code Evidence

Auto-installation from `save.py:853-906`:
```python
def install_llama_cpp_clone_non_blocking():
    full_command = [
        "git", "clone", "--recursive",
        "https://github.com/ggerganov/llama.cpp",
    ]
    run_installer = subprocess.Popen(
        full_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    return run_installer

def install_llama_cpp_make_non_blocking():
    check = os.system("make clean -C llama.cpp")
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

Environment detection from `save.py:76-81`:
```python
keynames = "\n" + "\n".join(os.environ.keys())
IS_COLAB_ENVIRONMENT = "\nCOLAB_" in keynames
IS_KAGGLE_ENVIRONMENT = "\nKAGGLE_" in keynames
KAGGLE_TMP = "/tmp"
```

## Related Pages

* [[requires_env::Implementation:unslothai_unsloth_save_pretrained_gguf]]
