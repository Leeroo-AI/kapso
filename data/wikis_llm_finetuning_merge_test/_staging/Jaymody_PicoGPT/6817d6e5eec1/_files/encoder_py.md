# File: `encoder.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 120 |
| Classes | `Encoder` |
| Functions | `bytes_to_unicode`, `get_pairs`, `get_encoder` |
| Imports | functools, json, os, regex |

## Understanding

**Status:** ✅ Explored

**Purpose:** BPE tokenizer for GPT-2 text encoding/decoding.

**Mechanism:** Implements byte-pair encoding (BPE) tokenization copied from OpenAI's GPT-2 repository. The `Encoder` class handles text-to-token conversion via BPE algorithm: `bytes_to_unicode()` creates a lookup table mapping UTF-8 bytes to unicode strings to avoid UNK tokens; `get_pairs()` extracts symbol pairs from words; the `bpe()` method iteratively merges the most frequent bigrams according to pre-trained merge rankings; `encode()` tokenizes text using regex patterns for contractions/words/numbers then applies BPE; `decode()` reverses the process. The `get_encoder()` factory loads encoder.json (vocab) and vocab.bpe (merge rules) from disk to instantiate an Encoder.

**Significance:** Core tokenization component - required for converting text prompts to token IDs that the GPT-2 model can process, and for decoding generated token IDs back to readable text.
