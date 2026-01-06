# File: `src/peft/tuners/oft/__init__.py`

**Category:** Core Adapter Implementation

| Property | Value |
|----------|-------|
| Lines | 52 |
| Imports | config, gptq, layer, model, peft |

## Understanding

**Status:** Fully explored

**Purpose:** Package initialization file that exports OFT (Orthogonal Finetuning) components and registers OFT as a PEFT method, with dynamic imports for quantized variants.

**Mechanism:**
1. Imports core OFT components: `OFTConfig`, `OFTLayer`, `OFTModel`, `GPTQOFTLinear`, and layer implementations (`Conv2d`, `Linear`)
2. Exports all classes through `__all__`
3. Registers OFT with PEFT framework:
   - Method name: "oft"
   - Config class: `OFTConfig`
   - Model class: `OFTModel`
4. Implements `__getattr__` for lazy loading of quantized variants:
   - `Linear8bitLt` (bitsandbytes 8-bit)
   - `Linear4bit` (bitsandbytes 4-bit)
   - `EetqOFTLinear` (EETQ quantization)

**Significance:** OFT (https://huggingface.co/papers/2306.07280) is a parameter-efficient fine-tuning method based on orthogonal transformations. Unlike LoRA which uses low-rank updates, OFT learns orthogonal matrices that preserve the Euclidean norm of activations. This provides theoretical benefits for stability and has shown strong empirical results. The implementation supports various quantization backends (bitsandbytes, GPTQ, AWQ, AQLM, EETQ, HQQ, INC) making it highly practical for memory-constrained scenarios.

## Key Components

- **Core Classes**: `OFTConfig`, `OFTLayer`, `OFTModel`, `Conv2d`, `Linear`, `GPTQOFTLinear`
- **Quantization Support**: Dynamic imports for 8-bit, 4-bit, EETQ variants
- **Method Registration**: Enables OFT as first-class PEFT adapter
- **Lazy Loading**: Quantized variants loaded on-demand to avoid import errors

## Quantization Backends

- **BitsAndBytes**: 8-bit and 4-bit Linear layers
- **GPTQ**: GPU-accelerated quantization
- **AWQ**: Activation-aware Weight Quantization
- **AQLM**: Additive Quantization of Language Models
- **EETQ**: Efficient Integer Quantization
- **HQQ**: Half-Quadratic Quantization
- **INC**: Intel Neural Compressor (FP8)
