# File: `src/transformers/quantizers/quantizer_gptq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 104 |
| Classes | `GptqHfQuantizer` |
| Imports | base, importlib, packaging, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements GPTQ (Accurate Post-Training Quantization) method, a popular weight-only quantization technique supporting both pre-quantized model loading and on-the-fly quantization.

**Mechanism:** Extends HfQuantizer by wrapping optimum's GPTQQuantizer for compatibility with Transformers API. Validates gptqmodel (>= 1.4.3) and optimum (>= 1.24.0) versions. For pre-quantized models, converts the model structure then post-initializes. For non-quantized models, performs full quantization using provided tokenizer and dataset. Enforces text-only model restriction and handles device mapping with CPU support via gptqmodel.

**Significance:** Widely-used quantizer leveraging the established GPTQ algorithm through optimum/gptqmodel libraries. Trainable and serializable, supporting both loading existing GPTQ models and creating new ones. Critical bridge between popular GPTQ ecosystem and Transformers framework.
