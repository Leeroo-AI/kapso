**Status:** ✅ Explored

**Purpose:** Implements FP-Quant quantization supporting both pseudo-quantization and real quantization using qutlass on Blackwell GPUs. Enables loading of prequantized models and in-flight quantization with support for quantization-aware training.

**Mechanism:** The FPQuantHfQuantizer class extends HfQuantizer and validates GPU/XPU availability, checks for qutlass (real quantization) or Triton (pseudo-quantization) backends, and replaces Linear layers with FPQuantLinear modules. It handles weight serialization/deserialization through FpQuantDeserialize and FpQuantQuantize operations, manages master weights for training, and enforces bfloat16 dtype. The quantizer processes models before weight loading by calling replace_with_fp_quant_linear() and supports parameter quantization for "weight", "qweight", and "dqweight" tensors.

**Significance:** Provides a flexible quantization approach supporting both emulated pseudo-quantization (no speedups but emulates behavior) for development/testing and real quantization for Blackwell GPUs with actual performance benefits. The support for QAT (quantization-aware training) when store_master_weights=True enables fine-tuning of quantized models, making it suitable for both inference optimization and continued training workflows.
