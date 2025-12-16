# File: `unsloth/utils/attention_dispatch.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 283 |
| Classes | `AttentionConfig`, `AttentionContext` |
| Functions | `select_attention_backend`, `run_attention` |
| Imports | __future__, dataclasses, models, torch, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a unified interface for dispatching attention operations to different backend implementations (FlashAttention, xFormers, or PyTorch SDPA) based on availability and performance characteristics.

**Mechanism:** Defines two dataclasses (AttentionConfig for per-layer metadata, AttentionContext for per-call info) and two main functions:
- `select_attention_backend()`: Returns optimal backend based on priority order (FlashAttention varlen/dense > xFormers > SDPA fallback)
- `run_attention()`: Executes attention with Q, K, V tensors using the selected backend, handling different tensor shapes, GQA (Grouped Query Attention), packed sequences, and causal masking for each backend
Supports four backend modes: FLASH_VARLEN (for variable-length packed sequences), FLASH_DENSE, XFORMERS, and SDPA. Each backend requires different tensor layouts and masking approaches, all handled internally.

**Significance:** Critical abstraction layer that enables Unsloth to leverage the fastest available attention implementation while maintaining a consistent API. Automatically handles complex details like GQA expansion, packed sequence metadata, sliding window attention, and backend-specific tensor reshaping. This flexibility is essential for maximizing performance across different hardware and installation configurations.
