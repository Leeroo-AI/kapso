# File: `src/peft/tuners/shira/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 129 |
| Classes | `ShiraConfig` |
| Imports | __future__, dataclasses, mask_functions, peft, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Configuration for SHiRA (Sparse High Rank Adapter), defining mask generation strategy and hyperparameters for sparse weight updates.

**Mechanism:** Extends PeftConfig with r (controls number of parameters via formula r*(m+n) to match LoRA), mask_type (default "random" for random sparse mask), random_seed for reproducibility, and standard PEFT parameters. In __post_init__, sets mask_fn to random_mask function or None, warning if unrecognized mask_type. Users can supply custom mask_fn by setting config.mask_fn = custom_function after instantiation.

**Significance:** SHiRA enables high-rank adaptation through structured sparsity. The parameter count r*(m+n) matches LoRA for fair comparison, but SHiRA trains sparse full-rank updates instead of dense low-rank updates. The mask_fn extensibility allows custom sparsity patterns beyond random (e.g., structured, learned masks). Custom mask functions must return binary masks matching base layer dimensions.
