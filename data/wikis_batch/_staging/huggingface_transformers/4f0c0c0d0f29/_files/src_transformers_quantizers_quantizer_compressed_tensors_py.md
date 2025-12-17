# File: `src/transformers/quantizers/quantizer_compressed_tensors.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 111 |
| Classes | `CompressedTensorsHfQuantizer` |
| Imports | base, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements quantization using the compressed-tensors library, which provides a unified framework for various compression techniques including quantization and sparsity. This quantizer handles both compressed and decompressed model states.

**Mechanism:** The `CompressedTensorsHfQuantizer` initializes a `ModelCompressor` from the quantization config and honors the `run_compressed` flag. During preprocessing, `apply_quantization_config()` sets up quantization wrappers, then conditionally calls `compress_model()` if quantization or sparsification is compressed. Post-loading, `decompress_model()` is called if needed for QAT scenarios. Suggests float16 dtype for efficiency. Updates tensor parallel plans for MoE expert layers with appropriate colwise/rowwise strategies. Fully trainable and serializable, with QAT support when not running in compressed mode.

**Significance:** Provides a flexible, modern quantization framework that goes beyond simple weight quantization to include sparsity and other compression techniques. The compress/decompress cycle enables both efficient storage and training workflows. Integration with tensor parallelism shows consideration for distributed training scenarios.
