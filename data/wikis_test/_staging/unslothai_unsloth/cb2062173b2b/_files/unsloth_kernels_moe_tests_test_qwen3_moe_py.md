# File: `unsloth/kernels/moe/tests/test_qwen3_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 273 |
| Functions | `model_id`, `config`, `annotated_context`, `test_qwen3_moe` |
| Imports | argparse, contextlib, grouped_gemm, moe_utils, pytest, torch, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** End-to-end integration tests for Qwen3 MoE blocks, validating torch grouped GEMM and Triton grouped GEMM implementations against HuggingFace reference with comprehensive intermediate result checking.

**Mechanism:** Implements thorough testing with multiple validation layers:

1. Test Infrastructure:
   - `annotated_context()`: Context manager for formatted test output sections
   - Pytest fixtures for model_id ("Qwen/Qwen3-30B-A3B") and config
   - Reduces autotune configs to 50 for manageable test runtime

2. Main Test (`test_qwen3_moe()`):
   - Parametrized across permute_x, permute_y, autotune, sequence lengths, dtypes
   - Tests against Qwen3-30B-A3B configuration (2048 hidden, 768 intermediate, 128 experts, top-8)
   - Three-way comparison:
     - HuggingFace `Qwen3MoeSparseMoeBlock` (reference)
     - `Qwen3MoeGroupedGEMMBlock` (torch-native grouped GEMM)
     - `Qwen3MoeFusedGroupedGEMMBlock` (Triton kernel grouped GEMM)

3. Forward Pass Validation (nested checks):
   - HF vs torch grouped GEMM: basic sanity check
   - Torch vs Triton: detailed intermediate result checking (token counts, routing indices, first/second GEMM outputs, unpermute states)
   - HF vs Triton: final correctness validation with verbose output

4. Backward Pass Validation (nested checks):
   - HF vs torch grouped GEMM: gradient sanity check
   - Torch vs Triton: gradient comparison for isolation
   - HF vs Triton: final gradient correctness with verbose output

5. Command-line Interface:
   - Allows standalone execution with configurable seqlen, dtype, permute_x, permute_y, autotune
   - Useful for debugging specific configurations

Important Note: Docstring warns that tests should be run as a module (python -m) NOT with pytest directly, due to random numerical errors from pytest/triton/autotuning interactions.

Tolerances: bfloat16 (1e-2), float16 (1e-3), float32 (1e-5)

**Significance:** Most comprehensive MoE integration test, validating not just final outputs but all intermediate computation steps. Critical for Qwen3's complex architecture (128 experts, top-8 routing, softmax-based routing with normalization). The three-layer validation strategy (HF→torch→Triton, torch→Triton, HF→Triton) provides multiple points of comparison to isolate issues precisely. Essential for ensuring that permutation fusion and Triton optimizations maintain correctness in a challenging high-expert-count scenario.
