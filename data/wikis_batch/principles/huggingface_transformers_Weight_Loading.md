{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Loading]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Weight loading is the process of materializing model parameters from checkpoint storage into executable memory representations while applying transformations for compatibility, efficiency, and device placement.

=== Description ===

After model architecture instantiation creates the structural skeleton, weight loading populates parameters with learned values from training. Modern weight loading is far more sophisticated than simple deserialization due to several challenges:

* '''Format Evolution:''' Model architectures evolve (e.g., fused vs. separate attention weights), requiring transformations between checkpoint formats and current code
* '''Memory Constraints:''' Large models cannot fit in single-device memory, requiring distributed loading and placement
* '''Quantization:''' Weights may be stored in full precision but loaded into quantized formats (or vice versa) for efficiency
* '''Device Heterogeneity:''' Models span GPUs, CPUs, and disk, with different layers on different devices
* '''Precision Mixing:''' Some layers (e.g., LayerNorm) must remain in float32 while others use float16/bfloat16

Weight loading must therefore:
1. Map checkpoint parameter names to model parameter names (handling renames)
2. Apply transformations (split QKV, merge experts, reshape)
3. Cast dtypes while respecting precision policies
4. Place parameters on target devices according to device maps
5. Apply quantization if configured
6. Track missing keys (parameters in model not in checkpoint) and unexpected keys (checkpoint params not used)

The process operates on symbolic (meta-device) parameters, replacing them with materialized tensors that have actual storage and device placement.

=== Usage ===

Use weight loading when:
* Implementing model deserialization from checkpoint files
* Building systems that support multiple checkpoint formats for the same model
* Creating pipelines that apply transformations during loading (quantization, precision changes)
* Designing distributed inference systems with heterogeneous device placement
* Developing model conversion tools that map between framework conventions

== Theoretical Basis ==

Weight loading implements a staged transformation pipeline:

'''Stage 1: Key Mapping and Collection'''
<pre>
INPUT: checkpoint_state_dict, model_meta_params, weight_mappings
OUTPUT: param_load_plan

# Build mapping from checkpoint keys to model parameter names
param_load_plan = {}

FOR EACH checkpoint_key IN checkpoint_state_dict.keys():
    # Apply renaming rules
    model_key = checkpoint_key
    FOR EACH renaming IN weight_mappings.renamings:
        model_key = renaming.apply(model_key)
    END FOR

    # Check for weight converters (split/merge operations)
    converter = find_matching_converter(model_key, weight_mappings.converters)

    IF converter IS NOT None THEN
        # This checkpoint key participates in a multi-key operation
        # e.g., qkv.weight -> q.weight, k.weight, v.weight
        collector = get_or_create_collector(converter)
        collector.add_source_tensor(model_key, checkpoint_state_dict[checkpoint_key])
    ELSE
        # Direct 1:1 mapping
        param_load_plan[model_key] = checkpoint_state_dict[checkpoint_key]
    END IF
END FOR

# Execute converters to generate target tensors
FOR EACH collector IN active_collectors:
    target_tensors = collector.execute_conversion()
    param_load_plan.update(target_tensors)
END FOR

RETURN param_load_plan
</pre>

'''Stage 2: Materialization and Placement'''
<pre>
INPUT: param_load_plan, model, device_map, dtype, quantizer
OUTPUT: model (modified in-place)

FOR EACH (target_key, source_tensor) IN param_load_plan:
    # Determine target device and dtype
    target_device = device_map.get(target_key, "cpu")
    target_dtype = dtype_plan.get(target_key, dtype)

    # Get reference to model parameter (currently on meta device)
    model_param = get_parameter(model, target_key)

    # Apply quantization if configured
    IF quantizer IS NOT None AND quantizer.should_quantize(target_key):
        materialized_tensor = quantizer.quantize(
            source_tensor,
            target_device=target_device,
            param_name=target_key
        )
    ELSE
        # Standard loading: cast and move
        materialized_tensor = source_tensor.to(
            device=target_device,
            dtype=target_dtype
        )
    END IF

    # Replace meta tensor with materialized tensor
    set_parameter(model, target_key, materialized_tensor)

    # Mark as initialized to prevent re-initialization
    materialized_tensor._is_hf_initialized = True

    # Track for diagnostics
    IF target_key IN model.state_dict() THEN
        missing_keys.remove(target_key)
    ELSE
        unexpected_keys.add(target_key)
    END IF
END FOR

RETURN model
</pre>

'''Stage 3: Validation and Finalization'''
<pre>
INPUT: model, missing_keys, unexpected_keys, mismatch_keys
OUTPUT: load_result

# Check for critical missing keys
critical_missing = missing_keys - model.ignorable_keys
IF critical_missing IS NOT EMPTY THEN
    WARN "Missing keys: {critical_missing}"
END IF

# Check for shape mismatches
IF mismatch_keys IS NOT EMPTY THEN
    WARN "Shape mismatches: {mismatch_keys}"
END IF

# Verify all parameters materialized
FOR EACH param IN model.parameters():
    ASSERT param.device != torch.device("meta"), f"Parameter {name} still on meta device"
END FOR

load_result = {
    "missing_keys": missing_keys,
    "unexpected_keys": unexpected_keys,
    "mismatched_keys": mismatch_keys
}

RETURN load_result
</pre>

'''Weight Conversion Examples:'''

'''Split (QKV attention):'''
<pre>
# Checkpoint has: attention.qkv.weight [3 * hidden_size, hidden_size]
# Model expects: attention.{q,k,v}.weight [hidden_size, hidden_size] each

Operation: Chunk(dim=0, chunks=3)
Input: qkv.weight [3072, 1024]
Output:
    q.weight [1024, 1024]
    k.weight [1024, 1024]
    v.weight [1024, 1024]
</pre>

'''Merge (MoE experts):'''
<pre>
# Checkpoint has: mlp.experts.0.weight, mlp.experts.1.weight, ..., mlp.experts.7.weight
# Model expects: mlp.experts_weight [8, hidden_size, ffn_size]

Operation: Stack(dim=0)
Input: 8 tensors of [hidden_size, ffn_size]
Output: mlp.experts_weight [8, hidden_size, ffn_size]
</pre>

'''Device Placement Strategies:'''
* '''Sequential:''' Layers 0-N on GPU 0, N+1 to M on GPU 1, etc.
* '''Balanced:''' Distribute layers to equalize memory usage across devices
* '''Balanced Low 0:''' Minimize GPU 0 usage (often reserved for user operations)
* '''Auto:''' Infer placement based on available memory and layer sizes

'''Memory Efficiency Techniques:'''
* '''Lazy Loading:''' Load tensors from disk as needed, not all at once
* '''Pinned Memory:''' Use CUDA pinned memory for faster CPU-GPU transfers
* '''Async Transfers:''' Overlap weight loading with computation
* '''Disk Offload:''' Keep infrequently-used parameters on disk, page in on demand

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_load_state_dict_in_model]]
