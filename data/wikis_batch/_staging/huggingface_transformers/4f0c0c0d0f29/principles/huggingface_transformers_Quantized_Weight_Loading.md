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

Loading pre-quantized weights or quantizing weights on-the-fly into the prepared quantized model structure.

=== Description ===

Quantized weight loading handles the actual population of the prepared model with numerical values. This involves:

* '''Loading''' quantized weights from checkpoints (for pre-quantized models)
* '''Quantizing''' standard weights during load (for runtime quantization)
* '''Unpacking''' compressed weight formats
* '''Reconstructing''' quantization metadata (scales, zero-points, codebooks)
* '''Distributing''' weights across devices according to device_map

The process differs significantly between pre-quantized models (loading already-quantized checkpoint) and runtime quantization (converting FP16/BF16 weights to quantized format during load).

=== Usage ===

This happens automatically during model loading after preprocessing. The quantizer's postprocess_model() method finalizes the loaded model, configuring runtime behavior and validating the quantized state.

== Theoretical Basis ==

=== Weight Reconstruction ===

For pre-quantized models, weights are stored in compressed format and must be reconstructed:

'''4-bit NF4 Storage:'''

Two 4-bit values packed per byte:
<math>
b = (w_1 \ll 4) | w_2, \quad w_1, w_2 \in [0, 15]
</math>

Unpacking:
<math>
w_1 = b \gg 4, \quad w_2 = b \& 0x0F
</math>

'''Dequantization:'''
<math>
W_{fp} = S \cdot \text{lookup}(W_{quant})
</math>

Where lookup maps quantized indices to their floating-point representations (e.g., NF4 lookup table).

=== Double Quantization Reconstruction ===

With nested quantization:

<math>
W = S_2 \cdot (S_{1,quant} \cdot W_{quant})
</math>

Loading process:
1. Load <math>W_{quant}</math> (4-bit packed)
2. Load <math>S_{1,quant}</math> (8-bit packed)
3. Load <math>S_2</math> (FP16/BF16)
4. Reconstruct: <math>S_1 = S_2 \cdot S_{1,quant}</math>

=== Weight Quantization On-the-Fly ===

For runtime quantization:

'''Compute Scale:'''
<math>
s = \frac{\max(|W|)}{2^{b-1} - 1}
</math>

'''Quantize:'''
<math>
W_{quant} = \text{clip}\left(\text{round}\left(\frac{W}{s}\right), -2^{b-1}, 2^{b-1} - 1\right)
</math>

'''For NF4:'''
<math>
W_{nf4} = \text{argmin}_{v \in \text{NF4\_table}} |W_i - s \cdot v|
</math>

=== Group-wise Quantization Loading ===

Weights are divided into groups:

<math>
W = [G_1, G_2, \ldots, G_k], \quad G_i \in \mathbb{R}^{n/k}
</math>

Each group has its own scale:

<math>
G_i = S_i \cdot G_{i,quant}
</math>

Loading must:
1. Load all <math>G_{i,quant}</math> (packed together)
2. Load all <math>S_i</math> (scale vector)
3. Reconstruct group-by-group

=== Checkpoint Format Variations ===

Different serialization formats:

'''safetensors Format:'''
* Separate tensors for weights, scales, metadata
* Keys: "weight", "weight.absmax", "weight.quant_state.*"

'''PyTorch Format:'''
* May bundle metadata with weights
* Requires unpacking nested structures

'''GGUF Format:'''
* Block-wise quantization
* Requires format-specific parser

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_quantizer_postprocess_model]]

=== Requires ===
* [[requires::Principle:huggingface_transformers_Quantized_Model_Preparation]]

=== Enables ===
* [[enables::Principle:huggingface_transformers_Quantized_Runtime_Optimization]]
