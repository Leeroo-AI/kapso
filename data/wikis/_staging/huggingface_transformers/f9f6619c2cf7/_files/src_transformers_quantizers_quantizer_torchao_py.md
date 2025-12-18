# File: `src/transformers/quantizers/quantizer_torchao.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 356 |
| Classes | `TorchAoHfQuantizer` |
| Functions | `fuzzy_match_size` |
| Imports | base, importlib, packaging, quantizers_utils, re, safetensors, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integrates PyTorch's torchao quantization library into transformers, supporting various quantization schemes including int4/int8 weight-only, dynamic activation quantization, and autoquant.

**Mechanism:** The `TorchAoHfQuantizer` class provides comprehensive torchao integration with sophisticated handling of multiple quantization types. It uses `fuzzy_match_size()` to extract bit-width from config names (e.g., "4weight" → int4), managing different serialization keys for each quantization scheme. The quantizer handles both pre-quantized model loading (with custom `TorchAoDeserialize` weight converters that reconstruct tensors from flattened safetensors format) and on-the-fly quantization (not requiring calibration). For autoquant, it compiles the model with torch.compile and applies automatic quantization selection. It manages complex device mapping scenarios including CPU/disk offload, adjusts memory calculations for quantized tensors (since torchao's internal representation doesn't report actual bit-width), and supports metadata extraction from safetensors. Version-dependent features are carefully handled through version checks (requires torchao ≥0.15.0 for serialization).

**Significance:** Provides the most feature-rich quantization integration in transformers, supporting PyTorch's official quantization library with multiple quantization schemes, training support for some configurations (int8), compilation compatibility, and sophisticated serialization. The autoquant mode offers automatic quantization strategy selection for optimal performance. This quantizer is crucial for PyTorch-native quantization workflows and represents the cutting edge of PyTorch quantization capabilities within the transformers ecosystem.
