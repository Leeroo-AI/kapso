{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Transforming model architecture to support quantized weights by replacing standard layers with quantization-aware implementations.

=== Description ===

Model preparation for quantization involves modifying the model's computational graph before weights are loaded. This process:

* '''Replaces''' standard Linear layers with quantized equivalents (e.g., Linear4bit, Linear8bit)
* '''Configures''' layers with quantization parameters
* '''Excludes''' specified modules from quantization
* '''Preserves''' model structure and connectivity

The preparation happens on the meta device (no actual weights), allowing efficient skeleton manipulation. Different quantization methods require different layer replacements and configurations.

=== Usage ===

This step is automatic during quantized model loading. It occurs after quantizer initialization but before weight loading. The model structure is modified in-place to accommodate quantized weights.

== Theoretical Basis ==

=== Layer Replacement Strategy ===

Standard layers are replaced with quantized equivalents:

<math>
L_{standard} \rightarrow L_{quant}
</math>

Where <math>L_{quant}</math> implements:

<math>
y = L_{quant}(x) = \text{dequant}(W_{quant}) \cdot x
</math>

=== Quantized Linear Layer ===

A quantized linear layer stores weights in compressed format:

'''Standard Linear:'''
<math>
W \in \mathbb{R}^{m \times n}, \text{ Memory: } m \cdot n \cdot 2 \text{ bytes (FP16)}
</math>

'''4-bit Quantized Linear:'''
<math>
W_{quant} \in \mathbb{Z}^{m \times n}_4, S \in \mathbb{R}^k, \text{ Memory: } \frac{m \cdot n}{2} + k \cdot 2 \text{ bytes}
</math>

Where:
* <math>W_{quant}</math> are 4-bit quantized weights
* <math>S</math> are scaling factors
* <math>k</math> is number of quantization groups

=== Module Selection Logic ===

Modules are quantized based on criteria:

<math>
\text{Quantize}(M) = \begin{cases}
\text{True} & \text{if } M \in \text{QuantizableTypes} \\
& \land M \notin \text{SkipModules} \\
& \land M \notin \text{OutputEmbeddings} \\
\text{False} & \text{otherwise}
\end{cases}
</math>

Typically:
* '''QuantizableTypes:''' Linear, Conv1d (method-dependent)
* '''SkipModules:''' User-specified exclusions
* '''OutputEmbeddings:''' lm_head, tied embeddings

=== Memory Device Handling ===

Preparation uses meta tensors for efficiency:

<math>
T_{meta}: \text{shape, dtype, device="meta"}
</math>

No actual memory is allocated, allowing fast skeleton manipulation. Weights are loaded later into the prepared structure.

=== Double Quantization Support ===

For methods supporting nested quantization:

<math>
W = S_2 \cdot S_1 \cdot W_{quant}
</math>

The preparation phase configures layers to handle nested scales.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_quantizer_preprocess_model]]

=== Requires ===
* [[requires::Principle:huggingface_transformers_Quantizer_Initialization]]

=== Enables ===
* [[enables::Principle:huggingface_transformers_Quantized_Weight_Loading]]
