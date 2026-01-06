# File: `tests/test_mixed.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 791 |
| Classes | `SimpleNet`, `TestMixedAdapterTypes` |
| Imports | copy, itertools, os, parameterized, peft, platform, pytest, re, tempfile, torch, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for mixed adapter types

**Mechanism:** Tests PeftMixedModel with multiple adapter types simultaneously (LoRA + AdaLora, LoHa + LoKr, etc.), including commutative/non-commutative combinations, adapter switching, and training behavior

**Significance:** Test coverage for using multiple different adapter types together
