"""
Evaluation script for CUDA multi-head attention optimization.

This script:
1. Loads the optimized module from the specified path
2. Checks numerical correctness against the baseline
3. Benchmarks both implementations
4. Reports speedup factor

Usage:
    python evaluate.py --path module.py

Based on: https://github.com/WecoAI/weco-cli/tree/main/examples/cuda
"""

import sys
import os
import shutil
import pathlib
import importlib
import importlib.util
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from triton.testing import do_bench


########################################################
# Baseline Implementation
########################################################
class Model(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    This is the baseline for correctness and performance comparison.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


########################################################
# Benchmark Utilities
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


def get_inputs(batch_size, seq_len, n_embd, device):
    """Generate random input tensor for benchmarking."""
    return torch.randn(batch_size, seq_len, n_embd, device=device, dtype=torch.float32)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CUDA attention optimization")
    parser.add_argument("--path", type=str, required=True, help="Path to the optimized module.py")
    args = parser.parse_args()

    # Setup local cache for PyTorch extensions
    cache_dir = pathlib.Path.cwd() / ".weco-temp/torch_extensions"
    shutil.rmtree(cache_dir.parent, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache_dir)

    # Benchmarking parameters
    n_correctness_trials = 10
    correctness_tolerance = 1e-5
    warmup_ms = 1e3
    rep_ms = 5 * 1e3

    # Model parameters
    max_seqlen = 512
    seq_len = 256
    n_embd = 768
    n_head = 8
    # Turn off dropout to measure correctness
    attn_pdrop = 0.0
    resid_pdrop = 0.0

    # Input parameters
    batch_size = 32

    print("=" * 60)
    print("CUDA Attention Optimization Evaluation")
    print("=" * 60)
    print(f"\nModel parameters:")
    print(f"  - max_seqlen: {max_seqlen}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - n_embd: {n_embd}")
    print(f"  - n_head: {n_head}")
    print(f"  - batch_size: {batch_size}")
    print()

    # Load solution module
    try:
        torch.manual_seed(0)
        solution_module = load_module_from_path(args.path, add_to_sys_modules=False)
        solution_model = solution_module.Model(
            n_embd=n_embd, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, max_seqlen=max_seqlen
        ).to("cuda")
        assert isinstance(solution_model, nn.Module)
        print(f"[OK] Loaded optimized model from: {args.path}")
    except Exception:
        print(f"[FAIL] Candidate module initialization failed:")
        print(traceback.format_exc())
        exit(1)

    # Load baseline model
    torch.manual_seed(0)
    baseline_model = Model(
        n_embd=n_embd, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, max_seqlen=max_seqlen
    ).to("cuda")
    print("[OK] Loaded baseline model")

    # Measure correctness
    print("\n--- Correctness Check ---")
    max_diff_avg = 0
    is_correct = True
    for trial in range(n_correctness_trials):
        inputs = get_inputs(batch_size=batch_size, seq_len=seq_len, n_embd=n_embd, device="cuda")
        with torch.no_grad():
            optimized_output = solution_model(inputs)
            if torch.isnan(optimized_output).any():
                print(f"[FAIL] NaN detected in optimized model output (trial {trial})")
                is_correct = False
                break
            if torch.isinf(optimized_output).any():
                print(f"[FAIL] Inf detected in optimized model output (trial {trial})")
                is_correct = False
                break
            baseline_output = baseline_model(inputs)
            max_diff_avg += torch.max(torch.abs(optimized_output - baseline_output))
    
    if is_correct:
        max_diff_avg /= n_correctness_trials
        print(f"Max float diff (avg over {n_correctness_trials} trials): {max_diff_avg:.2e}")
        if max_diff_avg > correctness_tolerance:
            print(f"[FAIL] Max diff {max_diff_avg:.2e} exceeds tolerance {correctness_tolerance:.2e}")
            is_correct = False
        else:
            print(f"[OK] Correctness check passed (tolerance: {correctness_tolerance:.2e})")

    # Measure performance
    print("\n--- Performance Benchmark ---")
    inputs = get_inputs(batch_size=batch_size, seq_len=seq_len, n_embd=n_embd, device="cuda")
    
    t_avg_baseline = do_bench(lambda: baseline_model(inputs), warmup=warmup_ms, rep=rep_ms)
    print(f"Baseline time: {t_avg_baseline:.2f} ms")
    
    t_avg_optimized = do_bench(lambda: solution_model(inputs), warmup=warmup_ms, rep=rep_ms)
    print(f"Optimized time: {t_avg_optimized:.2f} ms")
    
    speedup = t_avg_baseline / t_avg_optimized
    print(f"Speedup: {speedup:.2f}x")

    # Final score
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    if is_correct:
        print(f"Correctness: PASS")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Score: {speedup:.4f}")
    else:
        print(f"Correctness: FAIL")
        print(f"Score: 0.0 (incorrect solution)")

    # Clean up
    shutil.rmtree(cache_dir.parent, ignore_errors=True)

    # Return score for Kapso evaluation
    if is_correct:
        return speedup
    return 0.0


if __name__ == "__main__":
    score = main()
    print(f"\n__SCORE__: {score}")
