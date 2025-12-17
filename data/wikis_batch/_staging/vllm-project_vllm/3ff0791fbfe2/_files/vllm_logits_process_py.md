# File: `vllm/logits_process.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 121 |
| Classes | `NoBadWordsLogitsProcessor` |
| Functions | `get_bad_words_logits_processors` |
| Imports | collections, torch, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Logits processing for bad words

**Mechanism:** Implements NoBadWordsLogitsProcessor to prevent generation of prohibited words. Processes tokenized bad word sequences and applies negative infinity bias to logits for forbidden tokens. Handles both single-token and multi-token bad words, with prefix matching for multi-token sequences. The get_bad_words_logits_processors factory function tokenizes bad words with/without prefix spaces to handle tokenizer variations.

**Significance:** Enables content filtering and constrained generation by preventing specific words from being generated. Important for safety, compliance, and controlled generation scenarios. Provides the foundation for more complex constrained decoding features.
