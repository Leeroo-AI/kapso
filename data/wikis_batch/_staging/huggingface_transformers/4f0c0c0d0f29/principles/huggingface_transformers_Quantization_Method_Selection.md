{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|GPTQ|https://arxiv.org/abs/2210.17323]]
* [[source::Paper|AWQ|https://arxiv.org/abs/2306.00978]]
|-
! Domains
| [[domain::NLP]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Selecting an appropriate quantization method from available techniques based on target precision, hardware capabilities, and accuracy requirements.

=== Description ===

Quantization method selection involves choosing from multiple approaches to reduce model precision, each with different trade-offs. Methods include:

* '''BitsAndBytes (BnB):''' GPU-focused quantization supporting 8-bit (LLM.int8) and 4-bit (FP4/NF4) precision
* '''GPTQ:''' Post-training quantization to 2-4 bits using second-order information
* '''AWQ:''' Activation-aware weight quantization preserving important weights
* '''AQLM:''' Additive quantization using multiple codebooks
* '''Quanto:''' PyTorch-native quantization for various precisions
* '''EETQ:''' Efficient INT8 quantization
* '''HQQ:''' Half-quadratic quantization
* '''Compressed Tensors:''' Flexible quantization with multiple schemes
* '''FP8:''' 8-bit floating-point quantization for modern GPUs
* '''TorchAO:''' PyTorch's native quantization API
* '''BitNet:''' Extreme quantization to 1-bit
* '''SPQR:''' Sparse quantization with outlier handling

The selection determines the entire quantization workflow, configuration options, and runtime behavior.

=== Usage ===

Use when loading a quantized model or preparing to quantize a model. The method must align with available hardware accelerators (CUDA, ROCm, CPU) and the model's quantization state (pre-quantized vs. quantization-aware training).

== Theoretical Basis ==

=== Quantization Fundamentals ===

Quantization maps high-precision values to lower-precision representations:

<math>x_{quant} = \text{round}\left(\frac{x - z}{s}\right)</math>

Where:
* <math>x</math> is the original floating-point value
* <math>s</math> is the scale factor
* <math>z</math> is the zero-point (for asymmetric quantization)
* <math>x_{quant}</math> is the quantized integer value

=== Precision Levels ===

'''INT8 Quantization:'''
* 256 discrete levels
* Typically -128 to 127 for symmetric, 0 to 255 for asymmetric
* ~4x compression, moderate accuracy loss

'''INT4 Quantization:'''
* 16 discrete levels
* Range: -8 to 7 (signed) or 0 to 15 (unsigned)
* ~8x compression, higher accuracy loss

'''FP4/NF4 (4-bit Floating Point):'''
* NF4: Information-theoretically optimal for normally distributed weights
* Provides better dynamic range than INT4
* Used in QLoRA for efficient fine-tuning

'''FP8 (8-bit Floating Point):'''
* Hardware-accelerated on modern GPUs (H100, MI300)
* Maintains floating-point dynamic range
* E4M3 (4 exponent, 3 mantissa) or E5M2 format

=== Method Selection Criteria ===

'''Accuracy vs. Compression:'''
* Higher bits → better accuracy, less compression
* GPTQ/AWQ → optimize accuracy at given bit-width
* BnB → easy to use, good for inference

'''Hardware Support:'''
* CUDA kernels: BnB, GPTQ, AWQ, FP8
* CPU: Quanto, TorchAO with CPU layouts
* ROCm: Limited support, check method availability

'''Calibration Requirements:'''
* Post-training quantization (GPTQ, AWQ): requires calibration data
* Weight-only (BnB): no calibration needed
* Activation quantization: runtime calibration or static profiling

'''Model Modifications:'''
* Some methods replace Linear layers with custom implementations
* Others keep architecture intact with modified weights
* Consider serialization and deployment requirements

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_QuantizationMethod]]
