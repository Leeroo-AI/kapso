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

**Purpose:** Tokenizer loading, validation, and fixing utilities for consistent tokenization across models.

**Mechanism:**
- `load_correct_tokenizer()`: Main entry point that loads tokenizer with proper settings and fixes
- `try_fix_tokenizer()`: Repairs fast tokenizers by editing their JSON representation to fix special token mappings
- `convert_to_fast_tokenizer()`: Converts slow tokenizers to fast while preserving behavior
- `assert_same_tokenization()`: Validates slow and fast tokenizers produce identical outputs
- `fix_sentencepiece_tokenizer()`: Edits sentencepiece .model files to update token mappings
- `fix_sentencepiece_gguf()`: Extends tokenizer.model with added_tokens.json for GGUF export
- `fix_chat_template()`: Adds missing `{% if add_generation_prompt %}` blocks to chat templates
- `check_tokenizer()`: Validates tokenizer vocab size matches model embedding size, removes out-of-bounds tokens
- `patch_sft_trainer_tokenizer()`: Patches TRL SFTTrainer to handle BOS token duplication and untrained tokens

**Significance:** Critical for correct tokenization. Many models have broken tokenizers (wrong special tokens, missing chat templates, vocab mismatches). These utilities ensure reliable tokenization and prevent silent training corruption.
