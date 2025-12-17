# File: `unsloth/tokenizer_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1105 |
| Classes | `SentencePieceTokenTypes` |
| Functions | `try_fix_tokenizer`, `get_sorted_dict`, `convert_to_fast_tokenizer`, `assert_same_tokenization`, `fix_sentencepiece_tokenizer`, `fix_sentencepiece_gguf`, `load_correct_tokenizer`, `fix_chat_template`, `... +2 more` |
| Imports | collections, gc, inspect, itertools, numpy, os, peft, re, subprocess, torch, ... +3 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides comprehensive tokenizer utilities including conversion between slow/fast tokenizers, validation, fixing broken tokenizers, adding new tokens, and handling sentencepiece models.

**Mechanism:** Implements convert_to_fast_tokenizer() to migrate from slow Python tokenizers to fast Rust-based ones while maintaining compatibility. Validates tokenizer vocab and special tokens match original. Provides check_tokenizer() to detect and repair out-of-bounds token IDs. Handles SentencePiece protobuf manipulation for extending vocabularies. Patches SFTTrainer to handle tokenizer compatibility issues.

**Significance:** Solves real issues with Hugging Face tokenizers where fast and slow versions diverge, causing training failures. Enables adding custom tokens while maintaining model compatibility. Critical for smooth model loading without mysterious token ID errors.
