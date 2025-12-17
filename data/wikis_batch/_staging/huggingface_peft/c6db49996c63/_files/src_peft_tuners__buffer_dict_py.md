# File: `src/peft/tuners/_buffer_dict.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 159 |
| Classes | `BufferDict` |
| Imports | __future__, collections, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides ordered dictionary container for PyTorch buffers with proper module registration.

**Mechanism:** Implements BufferDict as a nn.Module subclass that holds buffers in an ordered dictionary-like interface, ensuring buffers are properly registered with PyTorch's module system and visible to state_dict operations. Adapted from torchbotorch implementation.

**Significance:** Utility class enabling adapter layers to store non-trainable state (buffers) in dictionary form with proper PyTorch integration, particularly useful for adapters that need to maintain multiple sets of buffers indexed by adapter names.
