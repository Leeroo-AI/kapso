"""
Evaluation script for PyTorch operation optimization.

This script:
1. Loads the optimized module from the specified path
2. Compares it against a baseline implementation
3. Checks numerical correctness (max diff < tolerance)
4. Benchmarks both implementations
5. Returns speedup as the score
"""

import time
import sys
import os
import pathlib
import importlib
import traceback
import json
import argparse
import torch
import torch.nn as nn


########################################################
# Baseline Implementation
########################################################
class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = torch.matmul(x, self.weight.T)
        x = x / 2
        x = torch.sum(x, dim=1, keepdim=True)
        x = x * self.scaling_factor
        return x


########################################################
# Module Loading
########################################################
def load_module_from_path(module_path: str, add_to_sys_modules: bool = False):
    """Load a Python module from a file path."""
    module_path = pathlib.Path(module_path)
    name = module_path.stem
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    if add_to_sys_modules:
        sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


########################################################
# Benchmark Function
########################################################
@torch.no_grad()
def bench(f, inputs, n_warmup, n_rep):
    """
    Benchmark a function with warmup and repetitions.
    
    Args:
        f: Function to benchmark
        inputs: Input tensor
        n_warmup: Number of warmup iterations
        n_rep: Number of benchmark repetitions
        
    Returns:
        Average time in milliseconds
    """
    device_type = inputs.device.type

    # Warmup
    for _ in range(n_warmup):
        f(inputs)
    if device_type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    t_avg = 0.0
    for _ in range(n_rep):
        start_time = time.time()
        f(inputs)
        t_avg += time.time() - start_time
        if device_type == "cuda":
            torch.cuda.synchronize()

    t_avg /= n_rep
    return t_avg * 1000  # Convert to milliseconds


def get_inputs(batch_size, input_size, device):
    """Generate random input tensor."""
    return torch.randn(batch_size, input_size, device=device, dtype=torch.float32)


########################################################
# Main Evaluation
########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to optimized module.py")
    parser.add_argument("--device", default="cuda", type=str, help="Device to run on (cuda/cpu)")
    args = parser.parse_args()

    # Benchmark parameters
    n_correctness_trials = 10
    correctness_tolerance = 1e-5
    n_warmup = 1000
    n_rep = 5000

    # Model parameters
    batch_size = 128
    input_size = 10
    hidden_size = 20
    scaling_factor = 1.5

    print("=" * 60)
    print("PyTorch Operation Optimization Evaluation")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Batch size: {batch_size}, Input size: {input_size}, Hidden size: {hidden_size}")
    print()

    # Load optimized module
    try:
        torch.manual_seed(0)
        solution_module = load_module_from_path(args.path, add_to_sys_modules=False)
        solution_model = solution_module.Model(input_size, hidden_size, scaling_factor).to(args.device)
        assert isinstance(solution_model, nn.Module), "Model must be an nn.Module"
        assert hasattr(solution_model, "forward"), "Model must have forward method"
        print(f"Loaded optimized module from: {args.path}")
    except Exception as e:
        print(f"FAILED: Could not load optimized module: {e}")
        print(traceback.format_exc())
        sys.exit(1)

    # Load baseline
    torch.manual_seed(0)
    baseline_model = Model(input_size, hidden_size, scaling_factor).to(args.device)

    # Correctness check
    print("\n--- Correctness Check ---")
    max_diff_avg = 0
    for trial in range(n_correctness_trials):
        inputs = get_inputs(batch_size, input_size, args.device)
        
        optimized_output = solution_model(inputs)
        if torch.isnan(optimized_output).any():
            print(f"FAILED: NaN detected in optimized output (trial {trial})")
            sys.exit(1)
        if torch.isinf(optimized_output).any():
            print(f"FAILED: Inf detected in optimized output (trial {trial})")
            sys.exit(1)
            
        baseline_output = baseline_model(inputs)
        max_diff = torch.max(torch.abs(optimized_output - baseline_output)).item()
        max_diff_avg += max_diff

    max_diff_avg /= n_correctness_trials
    print(f"Average max diff: {max_diff_avg:.2e}")
    
    if max_diff_avg > correctness_tolerance:
        print(f"FAILED: Max diff {max_diff_avg:.2e} exceeds tolerance {correctness_tolerance}")
        sys.exit(1)
    print(f"PASSED: Within tolerance ({correctness_tolerance})")

    # Performance benchmark
    print("\n--- Performance Benchmark ---")
    inputs = get_inputs(batch_size, input_size, args.device)
    
    t_baseline = bench(baseline_model, inputs, n_warmup, n_rep)
    print(f"Baseline time: {t_baseline:.4f} ms")
    
    t_optimized = bench(solution_model, inputs, n_warmup, n_rep)
    print(f"Optimized time: {t_optimized:.4f} ms")
    
    speedup = t_baseline / t_optimized
    print(f"Speedup: {speedup:.2f}x")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Correctness: PASS (max diff {max_diff_avg:.2e})")
    print(f"Baseline: {t_baseline:.4f} ms")
    print(f"Optimized: {t_optimized:.4f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Score: {speedup}")
