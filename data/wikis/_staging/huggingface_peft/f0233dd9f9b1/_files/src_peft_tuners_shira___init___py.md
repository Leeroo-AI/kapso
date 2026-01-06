# File: `src/peft/tuners/shira/__init__.py`

**Category:** initialization

| Property | Value |
|----------|-------|
| Lines | 28 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Package initialization file that exposes the SHiRA (Sparse High Rank Adapter) PEFT method with mixed-batch compatibility.

**Mechanism:**
- Imports and re-exports core SHiRA components: `ShiraConfig`, `ShiraLayer`, `ShiraModel`, and `Linear`
- Registers SHiRA with PEFT using `register_peft_method()` with:
  - `name="shira"`, `prefix="shira_"`
  - `is_mixed_compatible=True` (supports mixed adapter batches)
- Defines `__all__` for explicit export control

**Significance:** Essential initialization that makes SHiRA available as a PEFT method. SHiRA is a high-rank sparse adapter that achieves LoRA-equivalent parameter counts with full-rank adaptation by only updating a sparse subset of weight matrix elements. The sparse mask determines which elements are trainable, enabling parameter-efficient fine-tuning with potentially better expressiveness than low-rank methods. The method is particularly suitable when the adaptation requires capturing diverse patterns that low-rank factorization might miss.
