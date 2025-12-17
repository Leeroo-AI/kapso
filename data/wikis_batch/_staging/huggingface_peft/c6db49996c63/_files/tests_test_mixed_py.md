# File: `tests/test_mixed.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 791 |
| Classes | `SimpleNet`, `TestMixedAdapterTypes` |
| Imports | copy, itertools, os, parameterized, peft, platform, pytest, re, tempfile, torch, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for PeftMixedModel with multiple adapter types

**Mechanism:** Comprehensive test suite validating mixed adapters (LoRA, LoHa, LoKr, AdaLora) can be combined, merged, disabled, saved, and loaded correctly. Tests commutativity, parameter counting, and realistic decoder models with multiple adapters active simultaneously

**Significance:** Critical for ensuring the mixed adapter feature works correctly across different adapter types and configurations, enabling users to combine complementary adaptation strategies in a single model
