# Environment: vLLM Integration

## Category
Software/Inference

## Summary
Unsloth provides integration with vLLM for high-performance inference serving of fine-tuned models, particularly when exporting to GGUF format for deployment.

## Requirements

### Software Requirements
| Package | Version Constraint | Evidence |
|---------|-------------------|----------|
| vLLM | >= 0.4.0 | For GGUF model loading |
| PyTorch | >= 2.1.0 | CUDA backend requirement |
| transformers | >= 4.38.0 | Model architecture support |

### Hardware Requirements
| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU VRAM | 8GB | 24GB+ | Depends on model size and quantization |
| System RAM | 16GB | 32GB+ | For model loading |

### GGUF Quantization Methods
From `unsloth/save.py:104-131`:

| Method | Description | Use Case |
|--------|-------------|----------|
| `q8_0` | 8-bit quantization | Fast conversion, high quality |
| `q4_k_m` | 4-bit mixed precision | Recommended for production |
| `q5_k_m` | 5-bit mixed precision | Balance of size/quality |
| `q2_k` | 2-bit quantization | Maximum compression |
| `bf16` | Bfloat16 | Full precision, fast conversion |
| `f16` | Float16 | Full precision fallback |

## Integration Points

### GGUF Export for vLLM
From `save.py:1070-1335`:

```python
def save_to_gguf(
    model_name: str,
    model_type: str,
    model_dtype: str,
    is_sentencepiece: bool = False,
    model_directory: str = "unsloth_finetuned_model",
    quantization_method = "fast_quantized",
    first_conversion: str = None,
    is_vlm: bool = False,
    is_gpt_oss: bool = False,
):
```

### llama.cpp Integration
The GGUF conversion relies on llama.cpp for quantization:

```python
LLAMA_CPP_TARGETS = [
    "llama-quantize",
    "llama-cli",
    "llama-server",
]
```

## Environment Detection

From `save.py:77-81`:
```python
keynames = "\n" + "\n".join(os.environ.keys())
IS_COLAB_ENVIRONMENT = "\nCOLAB_" in keynames
IS_KAGGLE_ENVIRONMENT = "\nKAGGLE_" in keynames
```

Special handling for resource-constrained environments:
- Kaggle: 20GB disk limit, uses `/tmp` for larger models
- Colab: Memory management for GGUF conversion

## Deployment Workflow

1. Train model with Unsloth
2. Export to merged 16-bit format
3. Convert to GGUF with desired quantization
4. Serve with vLLM or llama.cpp

## Source Evidence

- GGUF Conversion: `unsloth/save.py:1070-1335`
- Quantization Methods: `unsloth/save.py:104-131`
- llama.cpp Integration: `unsloth/save.py:860-1024`

## Backlinks

[[required_by::Implementation:Unslothai_Unsloth_convert_to_gguf]]
[[required_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained_vllm]]
[[required_by::Implementation:Unslothai_Unsloth_GRPOTrainer_train]]

## Related

- [[Environment:Unslothai_Unsloth_Ollama]]
- [[Environment:Unslothai_Unsloth_CUDA_11]]
