# File: `tests/test_osf.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 72 |
| Classes | `DummyConfig`, `DummyModel` |
| Functions | `test_osf_roundtrip`, `test_osf_gradient_projection_hook`, `test_osf_merge_and_unload_and_unmerge_behavior` |
| Imports | peft, pytest, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for Orthogonal Subspace Fine-tuning (OSF) method

**Mechanism:** Tests OSF's weight decomposition/reconstruction roundtrip accuracy, gradient projection hook ensuring orthogonality constraints, and merge/unload behavior. Validates that gradients are properly projected to orthogonal subspaces during training

**Significance:** Ensures OSF method correctly constrains parameter updates to orthogonal subspaces of weight matrices, enabling efficient fine-tuning with reduced interference between learned features
