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

**Purpose:** IntermediateTensors equality testing

**Mechanism:** Tests IntermediateTensors.__eq__() method by comparing empty tensors, different keys, different values, same values, and different subclass instances. Validates that equality considers type, keys, and tensor content.

**Significance:** Ensures intermediate tensor caching and comparison works correctly in multi-stage model architectures, preventing subtle bugs in pipeline parallelism and speculative decoding.
