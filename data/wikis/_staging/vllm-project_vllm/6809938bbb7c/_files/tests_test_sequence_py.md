# File: `tests/test_sequence.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 49 |
| Classes | `AnotherIntermediateTensors` |
| Functions | `test_sequence_intermediate_tensors_equal` |
| Imports | torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** IntermediateTensors equality tests

**Mechanism:** Tests IntermediateTensors equality comparison including: different subclass types (should not be equal), empty tensors (should be equal), different keys (should not be equal), same keys with different tensor shapes (should not be equal), and same keys with same tensor values (should be equal).

**Significance:** Validates the equality comparison implementation for IntermediateTensors, which is used for passing hidden states between model layers. Important for distributed inference and debugging.
