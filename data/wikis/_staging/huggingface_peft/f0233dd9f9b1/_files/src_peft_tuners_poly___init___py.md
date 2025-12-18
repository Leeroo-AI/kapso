# File: `src/peft/tuners/poly/__init__.py`

**Category:** initialization

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Package initialization file that exposes the Poly (Polytropon) PEFT method to the PEFT library ecosystem.

**Mechanism:**
- Imports and re-exports the core Poly components: `PolyConfig`, `PolyLayer`, `PolyModel`, and `Linear`
- Registers the Poly method with PEFT using `register_peft_method()`, which integrates it into the library's adapter registry
- Defines `__all__` to control which names are exported when the module is imported

**Significance:** Core initialization file that makes Poly available as a PEFT adapter method. This file is essential for the Poly adapter to be discoverable and usable within the PEFT ecosystem. Poly (Polytropon) is a multi-task learning adapter that uses multiple LoRA modules combined with routing mechanisms for efficient parameter sharing across tasks.
