# File: `src/peft/tuners/randlora/__init__.py`

**Category:** initialization

| Property | Value |
|----------|-------|
| Lines | 41 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Package initialization file that exposes the RandLora PEFT method and provides lazy loading for quantized (BitsAndBytes) variants.

**Mechanism:**
- Imports and re-exports core RandLora components: `RandLoraConfig`, `RandLoraLayer`, `RandLoraModel`, and `Linear`
- Registers RandLora with PEFT using `register_peft_method()` with prefix "randlora_"
- Implements `__getattr__()` for lazy loading of quantized variants:
  - `Linear8bitLt`: Loaded when bitsandbytes 8-bit is available
  - `Linear4bit`: Loaded when bitsandbytes 4-bit is available
  - Raises AttributeError for undefined names
- Defines `__all__` for explicit export control

**Significance:** Essential initialization that makes RandLora available as a PEFT method with optional quantization support. RandLora is a memory-efficient adapter that uses random frozen projections with trainable diagonal scaling matrices, significantly reducing parameters compared to LoRA while maintaining performance. The lazy loading pattern ensures quantized variants are only imported when needed and when the required libraries are available.
