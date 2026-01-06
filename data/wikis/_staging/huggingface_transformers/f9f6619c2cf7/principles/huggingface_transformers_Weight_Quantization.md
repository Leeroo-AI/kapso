{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Quantization|https://huggingface.co/docs/transformers/main_classes/quantization]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|LLM.int8()|https://arxiv.org/abs/2208.07339]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Transform model architecture on the meta device to prepare for quantized weight loading by setting attributes and replacing placeholder modules.

=== Description ===
Weight quantization preprocessing addresses the challenge of applying quantization before weights are materialized. When models are initialized on PyTorch's meta device (zero memory footprint), this phase manipulates the model skeleton to mark it for quantization, potentially replace certain module types, and set attributes that will guide weight loading. The key insight is that structural changes to the model graph must happen before weights arrive, allowing the loading mechanism to place quantized tensors directly into the correct module types without expensive post-hoc conversion.

This principle separates architecture transformation from weight transformation. The model structure is modified to expect quantized weights, and metadata is attached to guide the loading process. For on-the-fly quantization backends (bitsandbytes), modules remain standard Linear layers with special flags. For pre-quantized formats (GPTQ, AWQ), Linear layers may be replaced with custom layer types that expect packed quantized weights.

=== Usage ===
Apply this principle when you need to:
* Initialize models with quantization-aware architecture before loading weights
* Mark specific modules for quantization while model is on meta device
* Replace standard layers with quantization-compatible equivalents
* Set model attributes that influence weight loading behavior
* Prepare device_map for quantized weight placement
* Convert module structures for pre-quantized weight formats

== Theoretical Basis ==

=== Meta Device Initialization ===

<pre>
PyTorch meta device allows creating model structure with zero memory:

with torch.device("meta"):
    model = LlamaForCausalLM(config)
    # Model structure exists, but no memory allocated for parameters
    # All parameters are "meta tensors" - shape without storage

This enables:
1. Inspecting model architecture without loading weights
2. Modifying module graph before weight materialization
3. Computing memory requirements before allocation
4. Replacing modules in-place without copying tensors
</pre>

=== Preprocessing Operations ===

<pre>
preprocess_model(model_on_meta, dtype):
    # Step 1: Mark model as quantized
    model.is_quantized = True
    model.quantization_method = config.quant_method

    # Step 2: Attach quantization config
    model.config.quantization_config = self.quantization_config

    # Step 3: If pre-quantized, convert module types
    if self.pre_quantized:
        self._convert_model_for_quantization(model)
        # Replaces torch.nn.Linear with custom quantized layers

    # Step 4: Backend-specific processing
    self._process_model_before_weight_loading(model, dtype=dtype)
    # Set module attributes, modify device_map, etc.

    return model
</pre>

=== Module Conversion for Pre-Quantized Weights ===

<pre>
Example: GPTQ quantization

Standard model:
model.layers[0].self_attn.q_proj: Linear(in=4096, out=4096, bias=False)
  ↓ preprocess_model
Quantized model:
model.layers[0].self_attn.q_proj: QuantLinear(in=4096, out=4096, bits=4, group_size=128)

QuantLinear expects:
- qweight: packed INT4 tensor (shape [in/8, out])
- qzeros: packed INT4 zero points
- scales: FP16 scales (shape [in/group_size, out])

Standard Linear.weight loader → QuantLinear custom loader
</pre>

=== Attribute Setting ===

<pre>
Quantization attributes guide loading:

model.is_quantized = True
# Triggers special code paths in weight loading

model.quantization_method = "bitsandbytes"
# Identifies which quantizer to use for postprocessing

module._is_quantized = True
# Marks specific module for quantization

module._quantization_config = config
# Provides module-level quantization parameters

These attributes are checked during weight loading:
if hasattr(module, "_is_quantized") and module._is_quantized:
    weight = load_and_quantize_weight(checkpoint_weight)
else:
    weight = load_weight_standard(checkpoint_weight)
</pre>

=== Device Map Adjustment ===

<pre>
Quantization affects memory requirements, so device_map must be updated:

Original model estimate: 14GB (FP32)
After preprocessing with 4-bit quantization: 3.5GB

preprocess_model may adjust device_map:
- Update memory estimates for quantized modules
- Place quantized layers on GPU
- Keep full-precision layers on CPU if needed
- Ensure quantization-compatible placement

Example:
if device_map == "auto":
    # Recalculate with quantized memory estimates
    device_map = get_balanced_memory(
        model,
        max_memory={0: "20GB"},  # GPU 0
        dtype=torch.float16,
        quantization_config=config,  # Factor in 4x reduction
    )
</pre>

=== On-the-Fly vs Pre-Quantized ===

<pre>
On-the-fly quantization (bitsandbytes):
1. Modules remain torch.nn.Linear
2. Set flags: module._is_hf_initialized = False
3. Weight loading: FP16 tensor → Linear layer
4. Postprocessing: Convert Linear → Linear8bitLt or Linear4bit

Pre-quantized (GPTQ, AWQ):
1. Replace torch.nn.Linear → QuantLinear
2. Set expected weight format
3. Weight loading: INT4 packed tensor → QuantLinear
4. No postprocessing needed (already quantized)

Trade-offs:
- On-the-fly: Flexible, slower loading
- Pre-quantized: Fast loading, requires quantized checkpoint
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Quantizer_preprocess]]
