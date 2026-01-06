# File: `src/peft/tuners/loha/__init__.py`

**Category:** Core Adapter Implementation

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** Fully explored

**Purpose:** Package initialization file that exports the LoHa (Low-Rank Hadamard Product) adapter components and registers the LoHa method with the PEFT framework.

**Mechanism:**
1. Imports core LoHa components: `LoHaConfig`, `LoHaLayer`, `LoHaModel`, and layer implementations (`Conv2d`, `Linear`)
2. Exports all LoHa classes through `__all__` for public API access
3. Registers LoHa as a PEFT method using `register_peft_method()` with:
   - Method name: "loha"
   - Configuration class: `LoHaConfig`
   - Model class: `LoHaModel`
   - Parameter prefix: "hada_" (Hadamard product)
   - Mixed compatibility flag: `True` (can be used in MixedModel)

**Significance:** This is the entry point for the LoHa adapter method in PEFT. LoHa is a parameter-efficient fine-tuning technique that uses Hadamard products of low-rank decompositions to adapt pre-trained models. The registration makes LoHa available as a tuning strategy throughout the PEFT ecosystem, enabling users to apply this efficient adaptation method with mixed adapter support.

## Key Components

- **Exported Classes**: `Conv2d`, `Linear`, `LoHaConfig`, `LoHaLayer`, `LoHaModel`
- **Method Registration**: Enables LoHa as a first-class PEFT adapter type
- **Mixed Compatibility**: Supports combining LoHa with other adapter types in a single model
