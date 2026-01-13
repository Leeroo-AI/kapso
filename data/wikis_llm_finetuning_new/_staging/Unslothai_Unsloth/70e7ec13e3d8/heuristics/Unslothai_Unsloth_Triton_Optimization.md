# Heuristic: Triton Kernel Optimization

## Category
Kernels/GPU_Optimization

## Summary
Unsloth leverages Triton for custom GPU kernels that provide significant performance improvements over standard PyTorch implementations. This heuristic guides optimization decisions for Triton-based operations.

## The Decision Framework

### When Triton Kernels Are Used

| Operation | Triton Kernel | Benefit |
|-----------|--------------|---------|
| RMSNorm | Yes | Fused operation, memory efficient |
| LayerNorm | Yes | Fused operation, memory efficient |
| RoPE Embedding | Yes | In-place rotation, cache-friendly |
| Cross-Entropy Loss | Yes | Memory-efficient backward pass |
| SwiGLU/GEGLU | Yes | Fused activation, reduced memory |
| FP8 Operations | Yes | Block-wise quantization support |

### Performance Characteristics

| Kernel Type | Memory Reduction | Speed Improvement |
|-------------|------------------|-------------------|
| Fused LayerNorm | ~50% | ~2x |
| Fused SwiGLU | ~30% | ~1.5x |
| Triton RoPE | ~40% | ~1.8x |
| FP8 Block Quant | ~75% | ~1.2x |

## Implementation Evidence

From `unsloth/kernels/` directory:

- `rms_layernorm.py` - Fused RMSNorm/LayerNorm kernels
- `rope_embedding.py` - Rotary Position Embedding kernels
- `cross_entropy_loss.py` - Memory-efficient CE loss
- `swiglu.py` - Fused SwiGLU activation
- `geglu.py` - Fused GEGLU activation
- `fp8.py` - FP8 quantization operations

### Triton Kernel Pattern

```python
@triton.jit
def kernel_function(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    # ... compute ...
    tl.store(output_ptr + offsets, result, mask=mask)
```

## Tribal Knowledge

### Best Practices

1. **Block Size Selection**: Power of 2, typically 128-1024
2. **Memory Coalescing**: Access patterns should be contiguous
3. **Shared Memory**: Use for reduction operations
4. **Warp-Level Operations**: Leverage `tl.num_warps` for parallelism

### Common Pitfalls

1. **Non-divisible dimensions**: May require padding or fallback
2. **Register Pressure**: High register usage can limit occupancy
3. **Memory Bandwidth**: Often the bottleneck, not compute

## Decision Tree

```
Start
  │
  ├─ Is operation fuseable?
  │   └─ Yes → Use Triton fused kernel
  │
  ├─ Is dimension divisible by block size?
  │   ├─ No → May need fallback to PyTorch
  │   └─ Yes → Optimal performance
  │
  └─ GPU compute capability >= 8.0?
      └─ Yes → Full Triton support
```

## Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `TRITON_CACHE_DIR` | Kernel cache location | `~/.triton/cache` |
| `UNSLOTH_USE_TRITON` | Enable Triton kernels | `1` |

## Source Evidence

- Kernel Implementations: `unsloth/kernels/*.py`
- Utility Functions: `unsloth/kernels/utils.py`
- Integration: `unsloth/models/*.py`

## Backlinks

[[used_by::Implementation:Unslothai_Unsloth_FP8_Kernels]]
[[used_by::Implementation:Unslothai_Unsloth_Flex_Attention]]
[[used_by::Implementation:Unslothai_Unsloth_GEGLU_Kernels]]
[[used_by::Implementation:Unslothai_Unsloth_LayerNorm_Kernel]]
[[used_by::Implementation:Unslothai_Unsloth_RMSNorm_Kernel]]
[[used_by::Implementation:Unslothai_Unsloth_SwiGLU_Kernel]]
[[used_by::Implementation:Unslothai_Unsloth_RoPE_Kernel]]

## Related

- [[Environment:Unslothai_Unsloth_CUDA_11]]
- [[Heuristic:Unslothai_Unsloth_Gradient_Checkpointing]]
