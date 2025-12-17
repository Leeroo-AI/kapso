**Status:** ✅ Explored

**Purpose:** Implements MXFP4 (Microscaling FP4) quantization using fbgemm kernels for 4-bit floating-point model compression. Supports both quantized model loading and on-the-fly quantization with automatic dequantization fallback for unsupported hardware.

**Mechanism:** The Mxfp4HfQuantizer class validates GPU compute capability (>=7.5 for CUDA, XPU for Intel), Triton version (>=3.4.0 for CUDA, >=3.5.0 for XPU), and kernels availability. It replaces modules with Mxfp4GptOssExperts through replace_with_mxfp4_linear() and handles tensor parallelism/expert parallelism plans. For prequantized models on unsupported hardware, it automatically dequantizes to bf16. The quantizer manages weight conversions through Mxfp4Deserialize/Mxfp4Dequantize operations and handles serialization by unswizzling data and reshaping blocks/scales. It tracks _blocks and _scales suffixes for quantized parameters.

**Significance:** MXFP4 enables aggressive 4-bit compression while maintaining model quality through microscaling precision techniques. The automatic fallback to dequantization on unsupported hardware ensures broad compatibility, while the specialized support for expert models (MoE architectures) with tensor/expert parallelism makes it particularly valuable for deploying large-scale mixture-of-experts models on modern accelerators like H100 and B200 GPUs.
