# File: `src/peft/tuners/ia3/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 330 |
| Classes | `IA3Layer`, `Linear`, `_ConvNd`, `Conv2d`, `Conv3d` |
| Imports | peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements IA3 layer classes that learn per-dimension scaling vectors to rescale activations for parameter-efficient fine-tuning, supporting Linear, Conv2d, and Conv3d layers.

**Mechanism:** IA3Layer is the base class storing ia3_l (learned scaling vectors) and is_feedforward flag. The update_layer() method initializes scaling vectors with shape (1, in_features) for feedforward or (out_features, 1) for non-feedforward, initialized to ones via reset_ia3_parameters(). Linear implements forward() that multiplies activations by scaling vectors: for feedforward, input is scaled before base layer (x * ia3_l); for non-feedforward, output is scaled after (output * ia3_l). Multiple active adapters are multiplicatively combined. The get_delta_weight() reconstructs equivalent weight deltas for merging. _ConvNd provides shared convolution logic, with Conv2d and Conv3d extending it for 2D/3D convolutions. Convolution layers reshape scaling vectors to match spatial dimensions and apply broadcasting.

**Significance:** These classes implement IA3's core algorithm (https://huggingface.co/papers/2205.05638): learning per-dimension scaling vectors instead of full weight matrices. For attention layers, output scaling modulates what information is passed forward. For feedforward layers, input scaling controls which features are amplified. This requires only d parameters per layer (one scalar per dimension) versus 2*d*r for LoRA rank r, achieving extreme parameter efficiency. The merge/unmerge operations enable switching between adapted and base model states. Support for convolution layers makes IA3 applicable to vision models beyond just transformers.
