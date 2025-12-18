# File: `src/transformers/tokenization_python.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1400 |
| Classes | `Trie`, `ExtensionsTrie`, `PythonBackend` |
| Imports | bisect, collections, tokenization_utils_base, typing, unicodedata, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the base class for Python-based (slow) tokenizers. Provides the Trie data structure for efficient added token splitting and the PythonBackend class that all slow tokenizers inherit from.

**Mechanism:** The `Trie` class implements prefix-tree matching to split text on added tokens in O(n) time. It uses a greedy longest-match algorithm during `split()`. The `PythonBackend` class manages added tokens via `_added_tokens_decoder` and `_added_tokens_encoder` dictionaries, handles special token addition, and provides `_encode_plus` for batch encoding. Uses `tokens_trie` to split text before tokenization, respecting added token properties (lstrip, rstrip, single_word). Implements dynamic special token handling via `special_tokens_pattern` (cls_sep, bos_eos, etc.).

**Significance:** Foundation for all slow tokenizers in transformers (BERT, GPT-2, RoBERTa, etc. when not using fast tokenizers). While fast tokenizers are preferred for speed, slow tokenizers remain important for models without fast tokenizer support, debugging, and as reference implementations. The Trie-based token splitting ensures added tokens are never split during tokenization.
