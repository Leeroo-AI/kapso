# Environment: huggingface_transformers_FlashAttention

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Flash Attention Integration|https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::GPU_Computing]], [[domain::Performance_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Environment for Flash Attention 2/3 acceleration providing 2-4x faster attention computation with reduced memory usage.

=== Description ===

This environment enables Flash Attention, a memory-efficient and faster implementation of the attention mechanism. Flash Attention reduces memory usage from O(N^2) to O(N) and provides significant speedups (2-4x) by fusing attention operations and using tiling to reduce HBM (GPU memory) access. It supports both Flash Attention 2 (Ampere+) and Flash Attention 3 (Hopper) implementations.

=== Usage ===

Use this environment for:
- **Long sequence training/inference**: Sequences > 2048 tokens
- **Memory-constrained scenarios**: When standard attention causes OOM
- **Performance-critical applications**: Reduce inference latency
- **Large batch processing**: Enable larger batch sizes

Activated by setting `attn_implementation="flash_attention_2"` in model loading.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux || Flash Attention not available on Windows/macOS
|-
| Hardware || NVIDIA GPU || Ampere or newer required
|-
| Compute Capability || >= 7.0 || Flash Attention 2 requires SM 7.0+
|-
| Compute Capability || >= 9.0 || Flash Attention 3 requires SM 9.0+ (Hopper)
|-
| CUDA || 11.6+ || Required for Flash Attention 2
|}

== Dependencies ==

=== Inherits from CUDA Environment ===

All dependencies from `huggingface_transformers_CUDA`, plus:

=== Required for Flash Attention 2 ===

* `flash-attn` >= 2.0 (pip install flash-attn)
* `triton` (dependency of flash-attn)
* `packaging` (for version checks)

=== Alternative Backends ===

* **NPU (Ascend)**: Native flash attention support via `torch_npu`
* **XPU (Intel)**: Flash attention via `intel_extension_for_pytorch`
* **SDPA**: PyTorch native `scaled_dot_product_attention` (fallback)

== Credentials ==

Same as CUDA environment. No additional credentials required.

== Quick Install ==

<syntaxhighlight lang="bash">
# Install Flash Attention 2 (requires CUDA 11.6+ and Ampere GPU)
pip install flash-attn --no-build-isolation

# Or for specific CUDA version
pip install flash-attn --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "from flash_attn import flash_attn_func; print('Flash Attention available')"
</syntaxhighlight>

== Code Evidence ==

Flash Attention availability check from `modeling_flash_attention_utils.py:49-55`:

<syntaxhighlight lang="python">
def is_flash_attn_available():
    return (
        is_flash_attn_3_available()
        or is_flash_attn_2_available()
        or is_torch_npu_available()
        or is_torch_xpu_available()
    )
</syntaxhighlight>

Lazy import with fallback from `modeling_flash_attention_utils.py:93-99`:

<syntaxhighlight lang="python">
if (implementation == "flash_attention_2" and is_fa2) or (implementation is None and is_fa2 and not is_fa3):
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
elif is_torch_npu_available():
    # Package `flash-attn` is unavailable on Ascend NPU
    from .integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
</syntaxhighlight>

Top-left mask support check from `modeling_flash_attention_utils.py:37-45`:

<syntaxhighlight lang="python">
def flash_attn_supports_top_left_mask():
    if is_flash_attn_3_available():
        return False
    if is_flash_attn_2_available():
        return not is_flash_attn_greater_or_equal_2_10()

    from .integrations.npu_flash_attention import is_npu_fa2_top_left_aligned_causal_mask
    return is_npu_fa2_top_left_aligned_causal_mask()
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: flash_attn not found` || Flash Attention not installed || `pip install flash-attn --no-build-isolation`
|-
|| `Compute capability < 7.0` || GPU too old for Flash Attention || Use `attn_implementation="sdpa"` instead
|-
|| `CUDA extension failed to load` || Compilation issues || Ensure CUDA toolkit matches PyTorch CUDA version
|-
|| `FlashAttention only supports fp16/bf16` || Wrong dtype || Use `torch_dtype=torch.float16` or `torch.bfloat16`
|}

== Compatibility Notes ==

* **Flash Attention 2**: Works on Ampere (A100, RTX 30xx) and newer
* **Flash Attention 3**: Requires Hopper (H100) architecture
* **Sliding Window**: Supported in Flash Attention 2.3+
* **GQA/MQA**: Grouped-query attention supported natively
* **NPU/XPU**: Alternative implementations available via vendor libraries
* **Fallback**: SDPA (`torch.nn.functional.scaled_dot_product_attention`) works on all GPUs

== Related Pages ==

* [[requires_env::Implementation:huggingface_transformers_PreTrainedModel_from_config]]
* [[requires_env::Implementation:huggingface_transformers_Pipeline_forward]]
