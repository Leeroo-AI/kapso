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

**Purpose:** Tokenizer loading, validation, repair, and chat template management

**Mechanism:** Provides: fast/slow tokenizer comparison and conversion with assertion checks, sentencepiece tokenizer fixes via protobuf editing, out-of-bounds token detection and repair, chat template validation and auto-fix for add_generation_prompt, BOS/EOS token handling, SFTTrainer patching for add_special_tokens, and multi-GPU checks with fix_untrained_tokens integration

**Significance:** Essential tokenizer infrastructure that ensures correct tokenization across model types (Llama, Mistral, Qwen, Phi, etc), prevents out-of-bounds errors from buggy tokenizers, and enables proper chat template functionality for instruction tuning
