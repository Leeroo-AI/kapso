# PyTorch Operation Optimization Example

This example demonstrates using Kapso to optimize a simple PyTorch model that performs matrix multiplication, division, summation, and scaling operations.

Based on: https://docs.weco.ai/examples/pytorch-optimization

## Goal

Optimize the `Model` class in `module.py` for maximum speedup while maintaining numerical correctness.

## Initial Implementation

The baseline model performs these operations sequentially:
1. Matrix multiplication: `x @ weight.T`
2. Division: `x / 2`
3. Summation: `sum(x, dim=1)`
4. Scaling: `x * scaling_factor`

## Optimization Opportunities

- **Operation fusion**: Combine multiple operations into fewer kernel launches
- **torch.compile()**: Use PyTorch's JIT compilation for automatic optimization
- **In-place operations**: Reduce memory allocations where safe
- **Custom kernels**: Write fused Triton/CUDA kernels for maximum performance

## Running the Example

```bash
# Activate the kapso conda environment
conda activate kapso

# Run the optimization
python run_evolve.py
```

## Evaluation

The evaluation script (`evaluate.py`) tests:
1. **Correctness**: Max float difference < 1e-5 over 10 trials
2. **Performance**: Speedup compared to baseline (higher is better)

## Expected Results

Typical optimizations can achieve 1.5-3x speedup by:
- Using `torch.compile()` for automatic kernel fusion
- Combining the division and scaling into a single multiplication
- Using `torch.einsum` for fused operations
