# PyTorch Operation Optimization Example

This example demonstrates using Kapso to optimize a simple PyTorch model that performs matrix multiplication, division, summation, and scaling operations.

## Environment Setup

Before running this example, install the required dependencies:

```bash
# Create and activate conda environment (recommended)
conda create -n kapso python=3.10
conda activate kapso

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Kapso from the project root
cd /path/to/kapso
pip install -e .
```

Authenticate the coding-agent CLIs the default config uses, then
verify everything with `kapso doctor`:

```bash
# Claude Code sessions (ideation, implementation): log in once
npm install -g @anthropic-ai/claude-code
claude auth login   # or: export CLAUDE_CODE_OAUTH_TOKEN=...

# Codex sessions (research, judging, utilities): log in once
npm install -g @openai/codex
codex login

# embeddings (memory and knowledge-search indexing)
export OPENAI_API_KEY=your_key_here

kapso doctor   # checks all of the above and prints fixes for misses
```

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
