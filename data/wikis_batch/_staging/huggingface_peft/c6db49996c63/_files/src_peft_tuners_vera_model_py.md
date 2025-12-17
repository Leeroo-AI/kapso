# File: `src/peft/tuners/vera/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 294 |
| Classes | `VeraModel` |
| Imports | __future__, _buffer_dict, config, layer, math, peft, torch, transformers, tuners_utils, typing, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main VeRA model class that adapts pretrained models with vector-based random matrix adaptation.

**Mechanism:** VeraModel extends BaseTuner and initializes shared projection matrices vera_A (r × max_in_features) and vera_B (max_out_features × r) once using Kaiming initialization with deterministic PRNG seed (projection_prng_key). _find_dim identifies maximum layer dimensions. _create_and_replace wraps target Linear/Conv1D layers with VeRA layers, supporting quantized variants via bitsandbytes. Ensures all adapters use same PRNG key and save_projection setting.

**Significance:** Orchestrates VeRA's core innovation: single pair of frozen random projections shared across all adapted layers, drastically reducing parameters compared to LoRA. Critical for multi-task scenarios where many adapters are loaded simultaneously, as projection sharing enables 10-100x parameter reduction.
