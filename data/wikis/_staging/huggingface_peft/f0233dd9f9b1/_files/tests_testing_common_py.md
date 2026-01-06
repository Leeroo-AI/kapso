# File: `tests/testing_common.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 1829 |
| Classes | `PeftCommonTester` |
| Imports | copy, dataclasses, diffusers, json, operator, os, packaging, peft, pickle, platform, ... +9 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for common PEFT model functionality

**Mechanism:** Provides PeftCommonTester class with comprehensive test methods for PEFT model operations including save/load, merging, adapter management, state dict operations, and various skip conditions for unsupported features across different config types

**Significance:** Test coverage for core PEFT functionality across all adapter types - central test infrastructure
