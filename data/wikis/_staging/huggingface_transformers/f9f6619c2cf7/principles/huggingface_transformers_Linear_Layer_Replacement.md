{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Quantization|https://huggingface.co/docs/transformers/main_classes/quantization]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Repo|bitsandbytes|https://github.com/TimDettmers/bitsandbytes]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Replace standard Linear modules with quantized equivalents that store weights in compressed format and perform dequantization during forward pass.

=== Description ===
Linear layer replacement addresses the core technical challenge of quantization: transforming modules that expect full-precision weights into modules that operate on compressed representations. Standard torch.nn.Linear modules store FP32/FP16 weight tensors and perform straightforward matrix multiplication. Quantized linear layers store weights as INT4/INT8 tensors alongside quantization metadata (scales, zero-points), and perform implicit dequantization during forward pass to compute outputs. This replacement must maintain the same mathematical interface while changing internal representation.

This principle handles both eager replacement (for pre-quantized models) and lazy replacement (for on-the-fly quantization). For pre-quantized models, standard Linear modules are replaced with quantized equivalents before weight loading, so checkpoint weights load directly into compressed format. For on-the-fly quantization, replacement happens after full-precision weights are loaded, converting them in-place to quantized representation.

=== Usage ===
Apply this principle when you need to:
* Convert full-precision Linear layers to quantized equivalents
* Support loading of pre-quantized model checkpoints
* Replace modules in-place while preserving model graph structure
* Maintain parent-child relationships in nested module hierarchies
* Handle special module types (e.g., MoE expert layers) that require custom treatment
* Ensure quantized layers maintain same input/output interface as originals

== Theoretical Basis ==

=== Linear Layer Interface ===

<pre>
Standard Linear layer:
class Linear(nn.Module):
    weight: Tensor[out_features, in_features]  # FP32/FP16
    bias: Tensor[out_features] | None

    def forward(x):
        return x @ weight.T + bias

Memory: out_features × in_features × 4 bytes (FP32)
Compute: FP32 matrix multiplication

Quantized Linear layer:
class QuantLinear(nn.Module):
    qweight: Tensor[...] packed INT4/INT8  # Compressed storage
    scales: Tensor[...]  # Per-channel or per-group scales
    zeros: Tensor[...] | None  # Zero points
    bias: Tensor[out_features] | None

    def forward(x):
        weight_fp16 = dequantize(qweight, scales, zeros)
        return x @ weight_fp16.T + bias

Memory: out_features × in_features ÷ 2 bytes (INT4)
Compute: Dequantize + FP16 matrix multiplication
</pre>

=== In-Place Module Replacement ===

<pre>
Replace module while preserving graph structure:

model:
  - layers[0]:
      - self_attn:
          - q_proj: Linear(4096, 4096)  # Replace this
          - k_proj: Linear(4096, 4096)
          - v_proj: Linear(4096, 4096)

Replacement algorithm:
1. Traverse model tree: for name, module in model.named_modules()
2. Check if replaceable: if isinstance(module, nn.Linear)
3. Create quantized equivalent: quant_linear = QuantLinear(...)
4. Get parent reference: parent, child_name = get_parent_and_name(name)
5. Replace in parent: setattr(parent, child_name, quant_linear)

Result:
  - layers[0]:
      - self_attn:
          - q_proj: QuantLinear(4096, 4096)  # Replaced
          - k_proj: QuantLinear(4096, 4096)
          - v_proj: QuantLinear(4096, 4096)

Parent's reference now points to QuantLinear
Forward pass automatically uses quantized version
</pre>

=== Module Type Mapping ===

<pre>
Different quantization backends use different layer types:

BitsAndBytes 4-bit:
torch.nn.Linear → bnb.nn.Linear4bit
- Stores weights in NF4/FP4 format
- Dequantizes to bfloat16 for computation

BitsAndBytes 8-bit:
torch.nn.Linear → bnb.nn.Linear8bitLt
- Stores weights in INT8 with outlier handling
- Mixed INT8 + FP16 computation

GPTQ:
torch.nn.Linear → auto_gptq.nn_modules.qlinear.QuantLinear
- Stores weights as packed INT4 tensors
- Group-wise quantization with scales

AWQ:
torch.nn.Linear → awq.modules.linear.WQLinear
- Activation-aware weight quantization
- Optimized INT4 kernels

Mapping:
QUANTIZATION_MODULE_MAP = {
    "bitsandbytes_4bit": bnb.nn.Linear4bit,
    "bitsandbytes_8bit": bnb.nn.Linear8bitLt,
    "gptq": auto_gptq.nn_modules.qlinear.QuantLinear,
    "awq": awq.modules.linear.WQLinear,
}
</pre>

=== Weight Format Conversion ===

<pre>
On-the-fly quantization (post-loading):
1. Load FP16 weights into standard Linear
2. Extract weight tensor: w_fp16 = linear.weight
3. Quantize: qweight, scales, zeros = quantize(w_fp16)
4. Create quantized layer: quant_linear = QuantLinear(...)
5. Set quantized params: quant_linear.qweight = qweight
6. Replace: parent._modules[name] = quant_linear

Pre-quantized (pre-loading):
1. Replace Linear → QuantLinear before loading
2. Load checkpoint: checkpoint contains qweight, scales, zeros
3. QuantLinear.load_state_dict() handles loading directly
4. No conversion needed

Trade-off:
- On-the-fly: Can quantize any FP16 checkpoint, slower
- Pre-quantized: Fast loading, requires quantized checkpoint
</pre>

=== Special Module Handling ===

<pre>
Some architectures have special modules requiring custom treatment:

Mixture of Experts (MoE):
- Standard: List of Linear experts
- Problem: Each expert is separate module, replacing individually is inefficient
- Solution: Replace entire expert list with sequential container

Example for Llama4TextExperts:
Before:
model.layers[0].block_sparse_moe.experts: ModuleList[
    0: Linear(4096, 14336),
    1: Linear(4096, 14336),
    ...
    7: Linear(4096, 14336)
]

After:
model.layers[0].block_sparse_moe.experts: SequentialLlama4TextExperts[
    0: QuantLinear(4096, 14336),
    1: QuantLinear(4096, 14336),
    ...
    7: QuantLinear(4096, 14336)
]

MODULES_TO_PATCH_FOR_QUANTIZATION = {
    "Llama4TextExperts": {
        "module_name": SequentialLlama4TextExperts,
        "quantization_methods": ["bitsandbytes", "gptq"],
    }
}
</pre>

=== Preservation of Module State ===

<pre>
When replacing, preserve relevant state:

old_linear = parent._modules[name]
new_quant_linear = QuantLinear(...)

# Copy non-weight state
new_quant_linear.bias = old_linear.bias  # If exists
new_quant_linear.training = old_linear.training
new_quant_linear._forward_hooks = old_linear._forward_hooks
new_quant_linear._backward_hooks = old_linear._backward_hooks

# Quantize and set weight
new_quant_linear.set_weight(quantize(old_linear.weight))

# Replace
parent._modules[name] = new_quant_linear

Ensures hooks, training mode, etc. are preserved
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Quantizer_convert_weights]]
