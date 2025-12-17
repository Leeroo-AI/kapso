**Status:** ✅ Explored

**Purpose:** Implements Half-Quadratic Quantization (HQQ) method supporting both prequantized model loading and on-the-fly quantization with training capability. HQQLinear layers replace nn.Linear modules with configurable per-module quantization settings.

**Mechanism:** The HqqHfQuantizer class validates environment and dtype requirements, then tags nn.Linear modules with quant_config through prepare_for_hqq_linear() before weight loading. It tracks HQQ-specific state_dict keys (excluding bias) and handles multi-GPU scenarios by patching the forward method to ensure device compatibility. After weight loading, it marks the model with is_hqq_quantized and is_hqq_serializable flags. The quantizer includes a compatibility hack that adds a dummy weight property to HQQLinear to prevent runtime errors when models access weight.dtype.

**Significance:** HQQ provides flexible quantization with per-module configuration support and trainability, making it suitable for both inference optimization and continued model fine-tuning. The serialization support and multi-GPU handling enable practical deployment scenarios, while the module-level quantization control allows selective quantization strategies for different model components, balancing accuracy and compression.
