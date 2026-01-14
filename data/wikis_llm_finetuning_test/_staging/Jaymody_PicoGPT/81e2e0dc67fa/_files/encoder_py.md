# File: `encoder.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 120 |
| Classes | `Encoder` |
| Functions | `bytes_to_unicode`, `get_pairs`, `get_encoder` |
| Imports | functools, json, os, regex |

## Understanding

**Status:** ✅ Explored

**Purpose:** BPE tokenizer implementation for GPT-2 text encoding/decoding.

**Mechanism:** Implements byte-pair encoding (BPE) tokenization copied from OpenAI's GPT-2 codebase. The `Encoder` class handles text-to-token and token-to-text conversion using: (1) `bytes_to_unicode()` to create a reversible mapping between UTF-8 bytes and unicode characters, (2) `get_pairs()` to find symbol pairs for BPE merging, (3) the `bpe()` method to iteratively merge character pairs based on pre-trained merge rankings, and (4) `encode()`/`decode()` methods for the full tokenization pipeline. Uses regex patterns to split text into tokens before BPE processing.

**Significance:** Core component - required for converting user prompts to token IDs that the model can process, and converting generated token IDs back to readable text. Used by both `gpt2.py` and `gpt2_pico.py` via `utils.py`.
