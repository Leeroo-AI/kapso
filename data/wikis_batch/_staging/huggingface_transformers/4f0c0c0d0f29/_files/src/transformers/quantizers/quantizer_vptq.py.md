**Status:** ✅ Explored

**Purpose:** Implements VPTQ (Vector Post-Training Quantization) method for loading prequantized models on CUDA GPUs. Requires pre-calibrated models and does not support training.

**Mechanism:** The VptqHfQuantizer class (requires_calibration=True) validates CUDA availability, accelerate library, and VPTQ>=0.0.4 package. Before weight loading, it determines modules_to_not_convert by merging quantization_config.modules_to_not_convert with model._keep_in_fp32_modules, then calls replace_with_vptq_linear() to replace Linear layers with VPTQ-specific quantized implementations. The quantizer provides no post-weight-loading processing and is marked as both serializable and non-trainable.

**Significance:** VPTQ focuses on vector-based post-training quantization techniques that require calibration for optimal compression. The GPU-only requirement and calibration dependency indicate this is designed for inference deployment scenarios with pre-calibrated models. The serialization support enables model distribution, while the non-trainable nature makes it specialized for production inference workloads where model weights are frozen after quantization.
