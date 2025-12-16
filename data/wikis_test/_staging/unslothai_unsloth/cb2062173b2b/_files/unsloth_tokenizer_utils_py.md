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

**Purpose:** Loads, validates, and fixes tokenizers to ensure correctness across slow/fast variants, SentencePiece models, and chat templates. Detects and repairs broken tokenizer configurations that cause out-of-bounds errors or incorrect tokenization.

**Mechanism:**
- **Tokenizer loading** (`load_correct_tokenizer`):
  - Tries slow tokenizer first (from_slow=True) for accurate reference
  - Loads fast tokenizer (Rust-based) for performance
  - Compares slow vs fast tokenization to verify equivalence
  - Falls back to slow-to-fast conversion if mismatch detected
- **Slow-to-fast conversion** (`convert_to_fast_tokenizer`):
  - Uses `convert_slow_tokenizer()` to get Rust tokenizer from Python version
  - Fixes token mismatches by patching tokenizer JSON string
  - Handles prepending issues (e.g., Llama's "▁" character)
  - Validates vocab, special tokens, and chat template equivalence
- **Tokenization validation** (`assert_same_tokenization`):
  - Tests special tokens, chat templates (Mistral, Llama), and mixed text
  - Catches edge cases like CodeLlama's unusual token handling
  - Validates add_generation_prompt behavior
- **Chat template fixing** (`fix_chat_template`):
  - Detects missing `{% if add_generation_prompt %}` blocks
  - Automatically patches templates by finding endfor/endif and appending generation prompt
  - Validates both HuggingFace and ShareGPT format support
- **SentencePiece fixes**:
  - `fix_sentencepiece_tokenizer`: Patches tokenizer.model with new tokens using protobuf manipulation
  - `fix_sentencepiece_gguf`: Extends SentencePiece vocab with added_tokens.json for GGUF compatibility
  - Handles user-defined tokens (type=4) vs control/normal tokens
- **Broken tokenizer repair** (`check_tokenizer`):
  - Detects out-of-bounds token IDs (e.g., Starling's <sep>=32002 exceeding vocab size)
  - Removes bad tokens from added_tokens_decoder/encoder
  - Attempts reload with slow tokenizer if fast version broken
  - Warns but continues if repair fails (better than crashing)
- **Special cases**: Ignores known issues (Mistral models, Phi-4, Qwen Coder without tool tokens)
- **TRL trainer patching** (`patch_sft_trainer_tokenizer`):
  - Injects BOS token detection logic into SFTTrainer
  - Prevents double-BOS when template already includes it
  - Patches train() method to call `fix_untrained_tokens()` and `fix_zero_training_loss()`
  - Validates gradient accumulation steps for known bugs

**Significance:** Tokenizer bugs are insidious - they often don't crash but silently produce wrong results (e.g., missing BOS leading to poor model performance). This module's validation catches these issues early. The slow-to-fast conversion is critical because HuggingFace's Rust tokenizers sometimes have subtle differences from Python versions. The out-of-bounds detection prevents memory corruption. Chat template fixing ensures models can actually generate (add_generation_prompt is required for inference). This defensive programming prevents countless hours of debugging for users who would otherwise train models with broken tokenizers.
