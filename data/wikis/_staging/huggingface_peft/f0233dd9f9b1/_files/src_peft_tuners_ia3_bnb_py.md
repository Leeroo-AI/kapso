# File: `src/peft/tuners/ia3/bnb.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 129 |
| Classes | `Linear8bitLt`, `Linear4bit` |
| Imports | layer, peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements IA3 layer variants for quantized models using bitsandbytes, enabling IA3 adaptation on 8-bit and 4-bit quantized linear layers for memory-efficient fine-tuning.

**Mechanism:** The file conditionally defines two classes when bitsandbytes is available. Linear8bitLt extends IA3Layer for 8-bit quantized layers (bnb.nn.Linear8bitLt), implementing forward() that applies IA3 scaling to quantized computations - for feedforward layers, inputs are scaled before the quantized layer; for non-feedforward, outputs are scaled after. Multiple adapters are multiplicatively combined. Linear4bit extends IA3Layer for 4-bit quantized layers (bnb.nn.Linear4bit), with similar logic but specialized for 4-bit operations. Both classes freeze the quantized base weights and only train the IA3 scaling vectors. The forward methods handle dtype conversions (float32 casting when needed) to ensure numerical stability with quantized operations.

**Significance:** These classes enable IA3 on quantized models, combining two orthogonal efficiency techniques: (1) quantization reduces base model memory by 4-8x through low-precision weights, and (2) IA3 reduces trainable parameters to only d scalars per layer. This combination is extremely powerful for low-resource scenarios - you can fine-tune a 70B parameter model in 4-bit quantization with only millions of trainable IA3 parameters. The classes handle the complexity of mixing quantized computation with full-precision adapter parameters, ensuring correct gradient flow and numerical stability.
