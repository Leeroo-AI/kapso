# File: `src/transformers/tokenization_utils_sentencepiece.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 316 |
| Classes | `SentencePieceBackend`, `SentencePieceExtractor` |
| Imports | convert_slow_tokenizer, os, shutil, tokenization_python, tokenization_utils_base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements slow tokenizer backend for SentencePiece-based models. Provides `SentencePieceBackend` class that loads and wraps `sentencepiece.SentencePieceProcessor` models, and `SentencePieceExtractor` for extracting vocabulary from .model files during conversions.

**Mechanism:** `SentencePieceBackend` initializes by loading a .model file via `spm.SentencePieceProcessor`. Optionally disables `add_dummy_prefix` for non-legacy mode by modifying the protobuf model. Overrides `_tokenize` to handle prefix spaces correctly (encodes `unk_token + text` then strips unk tokens). Uses `sp_model.piece_to_id` and `IdToPiece` for vocabulary conversions. The `_add_tokens` method checks if tokens exist in base vocab before adding to extended vocab. `SentencePieceExtractor` extracts vocab scores and generates merges using `generate_merges`.

**Significance:** Backend for slow tokenizers of SentencePiece models like LLaMA, T5, ALBERT, XLM-RoBERTa, and mBART when fast tokenizers are unavailable. While fast tokenizers are preferred, this provides fallback support and serves as reference implementation. The extractor enables converting between SentencePiece and other tokenizer formats.
