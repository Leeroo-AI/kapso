# Environment: unslothai_unsloth_llama_cpp

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|GGUF Format|https://github.com/ggerganov/ggml/blob/master/docs/gguf.md]]
|-
! Domains
| [[domain::Model_Export]], [[domain::Quantization]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

## Overview

llama.cpp installation environment for GGUF model conversion and quantization, required for local deployment with Ollama.

### Description

This environment provides the tooling needed to convert Unsloth-trained models to GGUF format for deployment with llama.cpp and Ollama. The conversion process:

1. **Merges LoRA adapters** into base model weights (16-bit float)
2. **Converts to GGUF format** using llama.cpp's `convert_hf_to_gguf.py`
3. **Quantizes** to various formats (q4_k_m, q8_0, etc.) using `llama-quantize`

The environment supports automatic installation of llama.cpp if not present, with compilation from source including CUDA support.

### Usage

Required for:
- `model.save_pretrained_gguf()` - Save model in GGUF format
- `model.push_to_hub_gguf()` - Upload GGUF to HuggingFace
- Ollama integration via Modelfile generation

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, macOS, or Windows || Linux recommended for best performance
|-
| RAM || 16GB+ || Required for model conversion; 32GB+ for 7B+ models
|-
| Disk || 50GB+ free space || Temporary space needed during conversion
|-
| Build Tools || gcc/clang, cmake, make || For compiling llama.cpp from source
|-
| Special || Kaggle has 20GB limit || Use /tmp directory for Kaggle environments
|}

## Dependencies

### System Packages

* `git` - For cloning llama.cpp
* `cmake` >= 3.14
* `gcc` >= 9.0 or `clang` >= 10.0
* `make` or `ninja`
* `cuda-toolkit` (optional, for GPU-accelerated quantization)

### Python Packages

* `gguf` (installed with llama.cpp python bindings)
* `numpy`
* `sentencepiece` (for tokenizer conversion)
* `safetensors` (for reading HF model weights)

### llama.cpp Binaries

The following binaries are compiled from llama.cpp source:
* `llama-quantize` - For quantizing GGUF files
* `llama-export-lora` - For exporting LoRA adapters
* `llama-cli` - For testing inference

## Credentials

No credentials required for local conversion. For pushing to HuggingFace:
* `HF_TOKEN`: HuggingFace write token

## Code Evidence

From `unsloth/save.py:69-74`:
```python
# llama.cpp specific targets - all takes 90s. Below takes 60s
LLAMA_CPP_TARGETS = [
    "llama-quantize",
    "llama-export-lora",
    "llama-cli",
]
```

From `unsloth/save.py:1979-1987`:
```python
except Exception as e:
    if IS_KAGGLE_ENVIRONMENT:
        raise RuntimeError(
            f"Unsloth: GGUF conversion failed in Kaggle environment.\n"
            f"This is likely due to the 20GB disk space limit.\n"
            f"Try saving to /tmp directory or use a smaller model.\n"
            f"Error: {e}"
        )
```

Quantization methods from `unsloth/save.py:104-131`:
```python
ALLOWED_QUANTS = {
    "q4_k_m": "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m": "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q8_0": "Fast conversion. High resource use, but generally acceptable.",
    "q2_k": "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    # ... more options
}
```

## Related Pages

* [[requires_env::Implementation:unslothai_unsloth_save_pretrained_gguf]]
