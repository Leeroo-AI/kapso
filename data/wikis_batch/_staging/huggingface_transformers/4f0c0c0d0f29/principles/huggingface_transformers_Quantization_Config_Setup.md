{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|LLM.int8|https://arxiv.org/abs/2208.07339]]
|-
! Domains
| [[domain::NLP]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Configuring quantization parameters including precision, data types, scaling strategies, and module-specific settings before model loading.

=== Description ===

Quantization configuration establishes all parameters that control how a model will be quantized. This includes:

* '''Precision selection:''' Choosing between 8-bit, 4-bit, or other precisions
* '''Data type specification:''' FP4, NF4, INT8, FP8, or other formats
* '''Compute dtype:''' The dtype used during computation (often higher than storage)
* '''Double quantization:''' Whether to quantize the quantization constants themselves
* '''Storage format:''' How quantized values are packed and stored
* '''Module exclusions:''' Which layers should remain in full precision
* '''Outlier handling:''' Thresholds and strategies for outlier values

The configuration is method-specific. For example, BitsAndBytesConfig handles bitsandbytes parameters while GPTQConfig handles GPTQ-specific options like group size and damping percent.

=== Usage ===

Use when preparing to load a quantized model or before quantizing a model. The configuration should be passed to `from_pretrained()` or saved with the model for later loading. Configuration choices significantly impact memory usage, inference speed, and accuracy.

== Theoretical Basis ==

=== Configuration Parameters ===

'''Precision and Data Type:'''

For integer quantization:
<math>
x_{int} = \text{clip}\left(\text{round}\left(\frac{x}{s}\right) + z, q_{min}, q_{max}\right)
</math>

For 4-bit: <math>q_{min} = -8, q_{max} = 7</math> (signed) or <math>q_{min} = 0, q_{max} = 15</math> (unsigned)

For 8-bit: <math>q_{min} = -128, q_{max} = 127</math> or <math>q_{min} = 0, q_{max} = 255</math>

'''NF4 (4-bit NormalFloat):'''

NF4 uses non-uniform quantization bins optimized for normally distributed weights:

<math>
\text{NF4} = \{-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, \ldots\}
</math>

These 16 levels are information-theoretically optimal for N(0,1) distribution.

'''Double Quantization:'''

Reduces overhead by quantizing the quantization constants (scales and zero-points):

<math>
s = s_2 \cdot s_1
</math>

Where <math>s_1</math> is quantized and <math>s_2</math> is the scale for <math>s_1</math>.

'''Compute vs. Storage Dtype:'''

* '''Storage:''' INT4, NF4 (compact representation)
* '''Compute:''' BF16, FP16, FP32 (dequantized for computation)

<math>
y = W_{bf16} \cdot x = \text{dequantize}(W_{int4}) \cdot x
</math>

This allows maintaining computational precision while reducing memory.

=== Outlier Management ===

'''LLM.int8() Threshold:'''

Values exceeding threshold are kept in FP16:

<math>
w_i = \begin{cases}
w_i^{fp16} & \text{if } |w_i| > \tau \\
\text{quantize}(w_i) & \text{otherwise}
\end{cases}
</math>

Default <math>\tau = 6.0</math> captures ~99.9% of normally distributed values.

=== Group-wise Quantization ===

Used by GPTQ, AWQ for better accuracy:

<math>
s_g = \frac{\max(|W_g|)}{2^{b-1} - 1}
</math>

Where <math>W_g</math> is a group of weights and <math>b</math> is bit-width. Smaller groups capture local statistics better but increase overhead.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_BitsAndBytesConfig]]

=== Requires ===
* [[requires::Principle:huggingface_transformers_Quantization_Method_Selection]]

=== Enables ===
* [[enables::Principle:huggingface_transformers_Quantizer_Initialization]]
