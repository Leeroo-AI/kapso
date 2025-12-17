# File: `tests/test_poly.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 100 |
| Classes | `TestPoly` |
| Imports | os, peft, tempfile, torch, transformers, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for Polytropon (Poly) multi-task adapter method

**Mechanism:** Validates Poly's task-specific skill routing using multiple LoRA modules combined via learned routing. Tests training with task_ids, loss reduction, generation, adapter disabling, and save/load functionality with multi-task seq2seq models

**Significance:** Ensures Poly method correctly learns task-specific combinations of skill modules, enabling efficient multi-task learning with parameter sharing across related tasks
