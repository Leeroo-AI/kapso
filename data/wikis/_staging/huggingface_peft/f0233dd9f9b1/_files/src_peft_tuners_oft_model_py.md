# File: `src/peft/tuners/oft/model.py`

**Category:** Core Model Implementation

| Property | Value |
|----------|-------|
| Lines | 199 |
| Classes | `OFTModel` |
| Imports | aqlm, awq, eetq, gptq, hqq, inc, layer, peft |

## Understanding

**Status:** Fully explored

**Purpose:** Implements `OFTModel` class that creates and manages Orthogonal Finetuning adapters with support for multiple quantization backends.

**Mechanism:**

### Class Attributes:
- **prefix**: "oft_" (parameter prefix)
- **tuner_layer_cls**: `OFTLayer`
- **target_module_mapping**: Default target modules per model architecture

### Key Method - `_create_and_replace()`:
1. Validates current_key is not None
2. Prepares kwargs with OFT-specific parameters:
   - r, oft_block_size, module_dropout
   - coft, eps, block_share
   - use_cayley_neumann, num_cayley_neumann_terms
   - fan_in_fan_out, init_weights
   - Quantization flags (8bit, 4bit)
3. Detects quantization configs (GPTQ, AQLM, AWQ)
4. Creates new `OFTLayer` or updates existing
5. Sets as non-trainable if not in active adapters

### Static Method - `_create_new_module()`:
Uses dispatcher pattern to select appropriate layer type:
1. **Dispatchers** (in order):
   - `dispatch_bnb_8bit` (8-bit quantization)
   - `dispatch_bnb_4bit` (4-bit quantization)
   - `dispatch_eetq` (EETQ quantization)
   - `dispatch_aqlm` (AQLM quantization)
   - `dispatch_awq` (AWQ quantization)
   - `dispatch_gptq` (GPTQ quantization)
   - `dispatch_hqq` (HQQ quantization)
   - `dispatch_inc` (Intel NC quantization)
   - `dispatch_default` (Standard Linear/Conv2d)
2. Returns first matching module type
3. Raises ValueError if no dispatcher matches

### Method - `_check_merge_allowed()`:
- Validates merge is supported (not GPTQ, not replicated layers)
- Raises ValueError if incompatible

**Significance:** OFTModel brings orthogonal transformations to PEFT with extensive quantization support. The dispatcher pattern elegantly handles the complexity of multiple quantization backends without tight coupling. This makes OFT practical for production where memory constraints often require quantization. The orthogonal constraint provides theoretical benefits (norm preservation, stability) while the implementation ensures compatibility across diverse hardware and quantization schemes.

## Key Features

- **Dispatcher Pattern**: Clean separation of quantization backends
- **Eight Quantization Types**: Comprehensive backend support
- **Orthogonal Transformations**: Norm-preserving adaptations
- **Merge Support**: Can merge adapters into base weights (except GPTQ)
- **Validation**: Checks compatibility before merge
- **Device Mapping**: Proper handling of distributed models

## Supported Layer Types

- **Standard**: torch.nn.Linear, torch.nn.Conv2d
- **Quantized**: All major quantization libraries supported

## References

- Paper: https://huggingface.co/papers/2306.07280
- Method: Orthogonal Finetuning
