# File: `src/transformers/tokenization_python.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1400 |
| Classes | `Trie`, `ExtensionsTrie`, `PythonBackend` |
| Imports | bisect, collections, tokenization_utils_base, typing, unicodedata, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements pure Python (slow) tokenizers that don't require the fast Rust-based tokenizers library. Provides the base class PythonBackend for all slow tokenizer implementations and the Trie data structure for efficient added token splitting.

**Mechanism:** PythonBackend extends PreTrainedTokenizerBase with Python-only tokenization logic. Uses a Trie (prefix tree) to efficiently split text on added/special tokens in O(n) time via the split method, which iterates through text tracking partial matches and performing greedy longest-match tokenization. ExtensionsTrie extends this with an extensions method for token autocompletion. Manages added tokens via _added_tokens_encoder/_added_tokens_decoder dictionaries and integrates with special token handling. Helper functions (_is_whitespace, _is_control, _is_punctuation) support character-level text processing in tokenizer implementations.

**Significance:** Provides fallback tokenization when the fast tokenizers library isn't available and serves as the base for implementing new tokenizers in pure Python. Essential for development/prototyping of new tokenization schemes, maintaining backward compatibility with older Python-only tokenizers, and supporting platforms where Rust compilation isn't feasible. The Trie implementation is particularly clever for handling the complex tokenization order when special tokens are present.
