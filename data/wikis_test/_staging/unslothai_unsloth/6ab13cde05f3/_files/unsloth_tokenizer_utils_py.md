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

**Purpose:** Tokenizer conversion and validation utilities

**Mechanism:** Converts slow tokenizers to fast, repairs sentencepiece tokenizers

**Significance:** Ensures consistent tokenization
