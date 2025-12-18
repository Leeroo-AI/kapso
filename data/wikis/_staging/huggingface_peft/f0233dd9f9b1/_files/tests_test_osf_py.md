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

**Purpose:** Tests for OSF (Orthogonal Subspace Finetuning) adapter

**Mechanism:** Tests OSF weight matrix decomposition/reconstruction roundtrip, gradient projection hooks for orthogonality, and merge/unload behavior

**Significance:** Test coverage for OSF adapter method
