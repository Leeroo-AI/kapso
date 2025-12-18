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

**Purpose:** Logit manipulation for constrained generation (bad words filtering).

**Mechanism:** Implements `LogitsProcessor` for blocking unwanted words during generation. `NoBadWordsLogitsProcessor` takes a list of bad word token sequences and masks their logits to negative infinity, preventing their selection. Handles both single-token and multi-token bad words by tracking sequence context. The processor checks if previous tokens match a bad word prefix and blocks the completing token. `get_bad_words_logits_processors()` factory function tokenizes bad words with and without prefix spaces to handle tokenizer variations. Validates token IDs against vocabulary size.

**Significance:** Enables content filtering and constrained generation by preventing specific words/phrases. Important for safety, brand guidelines, and controlled text generation. The multi-token support handles complex phrases, not just single words. This is part of vLLM's broader structured output and constrained decoding capabilities.
