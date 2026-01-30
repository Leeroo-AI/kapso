# CUDA Attention Optimization Example

This example demonstrates using Kapso's `evolve()` function to optimize a vanilla multi-head self-attention implementation for CUDA performance.

Based on: https://github.com/WecoAI/weco-cli/tree/main/examples/cuda

## Problem Description

The baseline implementation (`module.py`) is a standard PyTorch multi-head masked self-attention layer. The goal is to optimize it for maximum speedup while maintaining numerical correctness.

### Optimization Opportunities

1. **Flash Attention**: Use `torch.nn.functional.scaled_dot_product_attention` which implements memory-efficient attention
2. **Triton Kernels**: Write custom Triton kernels for fused operations
3. **Memory Optimization**: Reduce memory bandwidth by fusing QKV projection with attention
4. **Tensor Cores**: Leverage FP16/BF16 with tensor cores for faster matrix operations

### Constraints

- Must maintain the same `Model` class interface
- Must produce numerically correct results (max diff < 1e-5)
- Must work on CUDA devices

## Files

- `run_evolve.py` - Main script that uses Kapso to optimize the attention kernel
- `module.py` - Baseline implementation to be optimized
- `evaluate.py` - Evaluation script that checks correctness and measures speedup
- `requirements.txt` - Python dependencies

## Usage

### Run Kapso Evolution

```bash
cd examples/cuda_optimization
python run_evolve.py
```

This will:
1. Initialize Kapso
2. Run multiple iterations to find optimized implementations
3. Output the best solution to `./output/cuda_optimized`

### Manual Evaluation

To evaluate a specific implementation:

```bash
python evaluate.py --path module.py
```

## Expected Output

The evaluation script outputs:
- Correctness check (pass/fail)
- Baseline execution time
- Optimized execution time  
- Speedup factor (score)

Example:
```
============================================================
FINAL RESULTS
============================================================
Correctness: PASS
Speedup: 2.34x
Score: 2.3400
```

## Model Parameters

| Parameter | Value |
|-----------|-------|
| max_seqlen | 512 |
| seq_len | 256 |
| n_embd | 768 |
| n_head | 8 |
| batch_size | 32 |

## Success Criteria

- **Correctness**: Max float difference < 1e-5
- **Performance**: Speedup > 1.0x (higher is better)

A good optimization should achieve 2-4x speedup using Flash Attention or custom Triton kernels.
