# File: `src/peft/tuners/lora/dora.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 203 |
| Classes | `DoraLinearLayer`, `DoraEmbeddingLayer`, `_DoraConvNdLayer`, `DoraConv1dLayer`, `DoraConv2dLayer`, `DoraConv3dLayer` |
| Imports | copy, peft, torch |

## Understanding

**Status:** ✅ Documented

**Purpose:** DoRA (Weight-Decomposed Low-Rank Adaptation) layer implementations

**Mechanism:** Decomposes weight updates into magnitude and direction components: W = m * (W0 + BA) / ||W0 + BA||. Implements specialized layers for Linear, Embedding, and Conv1d/2d/3d that maintain separate magnitude vectors and apply directional LoRA updates, with efficient CPU offloading support for memory management.

**Significance:** Important LoRA variant that improves training stability and final performance by explicitly modeling weight magnitude changes. Particularly effective for vision tasks and scenarios requiring precise control over weight update dynamics, at the cost of slightly increased computation.
