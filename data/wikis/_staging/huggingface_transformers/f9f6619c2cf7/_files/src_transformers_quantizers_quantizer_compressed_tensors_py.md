# File: `src/transformers/quantizers/quantizer_compressed_tensors.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 111 |
| Classes | `CompressedTensorsHfQuantizer` |
| Imports | base, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements quantization support for the compressed-tensors library, enabling loading and saving of models quantized using various compression schemes including quantization and sparsification.

**Mechanism:** Extends HfQuantizer base class to integrate with the compressed-tensors package's ModelCompressor. Before weight loading, applies quantization configuration and optionally compresses the model. After loading, decompresses if needed (especially for QAT). Supports both compressed and uncompressed runtime modes via run_compressed flag. Implements tensor parallelism plan updates for MoE models and provides serialization support.

**Significance:** Core quantizer enabling compressed-tensors integration, which supports advanced compression techniques including both quantization and sparsification. Unique in supporting quantization-aware training (QAT) and being both trainable and serializable, making it suitable for training workflows.
