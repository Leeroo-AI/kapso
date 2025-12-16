# File: `unsloth/kernels/moe/tests/test_qwen3_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 273 |
| Functions | `annotated_context`, `test_qwen3_moe` |
| Imports | argparse, contextlib, grouped_gemm, moe_utils, pytest, torch, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** End-to-end test suite for Qwen3 MoE block implementations, validating torch reference and Triton-optimized versions against HuggingFace.

**Mechanism:**

**Test setup**:
- Loads Qwen3-30B-A3B config from HuggingFace
- Creates three implementations:
  - HF reference (Qwen3MoeSparseMoeBlock)
  - Torch grouped GEMM (Qwen3MoeGroupedGEMMBlock)
  - Triton optimized (Qwen3MoeFusedGroupedGEMMBlock)
- Verifies weight equivalence across implementations

**Validation stages**:
1. Forward: HF vs torch grouped GEMM
2. Forward: torch vs Triton (with intermediate result checks)
3. Forward: HF vs Triton (main test)
4. Backward: HF vs torch grouped GEMM
5. Backward: torch vs Triton
6. Backward: HF vs Triton (main test)

**Intermediate validation**:
- Checks first_gemm, intermediate, second_gemm, hidden_states_unpermute
- Critical for isolating which stage introduces errors

**Test parameters**:
- Parametrized over seq_len, dtype, permute_x, permute_y, autotune
- Tests both manual and autotuned configurations

**Note**: Includes warning about pytest interaction with Triton autotuning causing non-deterministic failures. Recommends running as module instead of pytest for reliable results.

**Significance:** Comprehensive integration test for Qwen3 MoE with topk=8 (vs topk=1 for Llama4). Tests both permutation fusions (permute_x and permute_y) which are critical for Qwen3's dense-sparse mixture architecture. Essential for validating correctness on models with high expert counts.