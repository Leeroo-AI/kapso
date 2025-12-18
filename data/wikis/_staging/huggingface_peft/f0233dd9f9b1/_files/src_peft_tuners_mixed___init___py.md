# File: `src/peft/tuners/mixed/__init__.py`

**Category:** Mixed Adapter Framework

| Property | Value |
|----------|-------|
| Lines | 18 |
| Imports | model |

## Understanding

**Status:** Fully explored

**Purpose:** Package initialization file that exports the MixedModel class, enabling users to combine multiple different PEFT adapter types in a single model.

**Mechanism:**
1. Imports `COMPATIBLE_TUNER_TYPES` constant and `MixedModel` class from model module
2. Exports both through `__all__` for public API access
3. Provides entry point for mixed adapter functionality without direct registration (mixed adapters are created via `get_peft_model(mixed=True)`)

**Significance:** This is the entry point for PEFT's mixed adapter functionality, which allows combining different adapter types (LoRA, LoHa, LoKr, AdaLoRA, OFT, Shira) in a single model. This enables sophisticated multi-adapter strategies where different layers or modules use different adaptation techniques optimized for their specific roles. The mixed model approach provides maximum flexibility for advanced PEFT users who want to leverage the strengths of multiple methods simultaneously.

## Key Components

- **COMPATIBLE_TUNER_TYPES**: Tuple defining which adapter types can be mixed
- **MixedModel**: Main class for managing multiple adapter types
- **Usage Pattern**: Created via `get_peft_model(model, config, mixed=True)`
