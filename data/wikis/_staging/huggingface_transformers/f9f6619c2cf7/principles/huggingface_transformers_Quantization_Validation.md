{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Quantization|https://huggingface.co/docs/transformers/main_classes/quantization]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Software_Engineering]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Validate runtime environment compatibility with quantization requirements before loading model weights to prevent silent failures or resource conflicts.

=== Description ===
Quantization validation addresses the critical challenge of detecting incompatibilities between quantization requirements and the execution environment before expensive operations begin. Different quantization backends have specific dependencies (library versions, hardware capabilities), and certain configurations conflict with other loading options (device mapping strategies, dtype specifications, low_cpu_mem_usage flags). This principle establishes a pre-flight check mechanism that fails fast with actionable error messages rather than allowing silent failures or cryptic errors during weight loading.

The validation phase occurs after quantizer selection but before any model initialization, allowing the system to abort early if prerequisites aren't met. This includes checking for library availability, CUDA compute capability, device map compatibility, and parameter conflicts.

=== Usage ===
Apply this principle when you need to:
* Verify quantization library dependencies before model loading
* Check GPU capabilities for specific quantization methods
* Validate device_map compatibility with quantization strategies
* Detect conflicting parameters (e.g., load_in_8bit + torch_dtype)
* Provide early, clear error messages for misconfiguration
* Prevent resource exhaustion from incompatible configurations

== Theoretical Basis ==

=== Validation Categories ===

<pre>
1. Dependency Validation:
   - Library availability: is_bitsandbytes_available()
   - Version requirements: bitsandbytes >= 0.39.0
   - CUDA availability: torch.cuda.is_available()

2. Hardware Validation:
   - Compute capability: capability >= (7, 5) for INT8
   - GPU memory: sufficient for quantized model + overhead
   - Multi-GPU compatibility: device_map strategy

3. Configuration Validation:
   - Mutual exclusivity: not (load_in_8bit and load_in_4bit)
   - Parameter conflicts: load_in_8bit implies torch_dtype not specified
   - Device map compatibility: "auto" required for multi-GPU quantization

4. Model Architecture Validation:
   - Supported layer types: Linear, Conv2d
   - Unsupported features: certain attention mechanisms
   - Module naming conventions: for skip_modules functionality
</pre>

=== Validation Sequence ===

<pre>
validate_environment(device_map, weights_only, **kwargs):
    # Step 1: Backend availability
    if not backend_available():
        raise ImportError(f"pip install {backend_name}")

    # Step 2: Hardware requirements
    if cuda_required and not torch.cuda.is_available():
        raise RuntimeError("Quantization requires CUDA")

    if min_compute_capability:
        actual = get_compute_capability()
        if actual < min_compute_capability:
            raise RuntimeError(f"Need compute capability {min_compute_capability}, got {actual}")

    # Step 3: Parameter conflicts
    if quantization_enabled and dtype_specified:
        raise ValueError("Cannot specify torch_dtype with quantization")

    if quantization_enabled and device_map is None:
        raise ValueError("Must provide device_map for quantization")

    # Step 4: Device map validation
    if multi_gpu_required and device_map not in ["auto", "balanced"]:
        raise ValueError("Multi-GPU quantization requires device_map='auto'")

    # Step 5: Backend-specific checks
    backend.validate_specific_requirements(kwargs)
</pre>

=== Error Message Design ===

Good validation provides actionable feedback:

<pre>
# Bad error:
"Quantization failed"

# Good error:
"BitsAndBytes quantization requires CUDA compute capability >= 7.5.
Your GPU (GeForce GTX 1060) has compute capability 6.1.
Consider using GPTQ quantization instead, or upgrading your GPU."

# Include:
1. What failed (specific check)
2. Why it failed (actual vs. expected)
3. How to fix it (actionable steps or alternatives)
</pre>

=== Validation vs. Fallback Strategies ===

<pre>
Strict validation (preferred):
if not compatible:
    raise Error("Not compatible")
# Fails fast, clear error

Silent fallback (dangerous):
if not compatible:
    return original_model  # User may not notice quantization didn't apply
# Silent failure, confusing behavior

Warned fallback (sometimes appropriate):
if not compatible:
    warnings.warn("Falling back to FP16")
    return fp16_model
# Use only when fallback is explicitly designed behavior
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Quantizer_validate_environment]]
