# File: `unsloth/tokenizer_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1106 |
| Classes | `SentencePieceTokenTypes` |
| Functions | `try_fix_tokenizer`, `get_sorted_dict`, `convert_to_fast_tokenizer`, `assert_same_tokenization`, `fix_sentencepiece_tokenizer`, `fix_sentencepiece_gguf`, `load_correct_tokenizer`, `fix_chat_template`, `... +2 more` |
| Imports | collections, gc, inspect, itertools, numpy, os, peft, psutil, re, subprocess, ... +4 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tokenizer loading, fixing, and conversion utilities to handle quirks in various tokenizer implementations (especially SentencePiece-based models).

**Mechanism:** Key functions include `try_fix_tokenizer()` for automatic repairs, `convert_to_fast_tokenizer()` for slow-to-fast conversion, `fix_sentencepiece_tokenizer()` and `fix_sentencepiece_gguf()` for SentencePiece-specific issues, `load_correct_tokenizer()` for reliable loading, and `fix_chat_template()` for template repairs. The `SentencePieceTokenTypes` class provides token type constants. Handles edge cases like missing BOS/EOS tokens, incorrect vocab sizes, and broken chat templates.

**Significance:** Core component - ensures tokenizers work correctly across different model architectures. Critical for training data preprocessing and model inference, as tokenizer bugs can silently corrupt training.
