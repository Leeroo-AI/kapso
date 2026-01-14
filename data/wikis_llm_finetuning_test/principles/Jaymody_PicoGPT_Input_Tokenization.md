# Principle: Input_Tokenization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Neural Machine Translation of Rare Words with Subword Units|https://arxiv.org/abs/1508.07909]]
* [[source::Blog|Byte Pair Encoding|https://huggingface.co/learn/nlp-course/chapter6/5]]
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::Text_Processing]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==

Algorithm for converting raw text strings into sequences of integer token IDs using Byte Pair Encoding (BPE).

=== Description ===

Input Tokenization is the process of segmenting text into discrete units (tokens) that can be processed by a neural network. GPT-2 uses Byte Pair Encoding (BPE), a subword tokenization algorithm that:

1. **Avoids out-of-vocabulary (OOV) tokens** by operating at the byte level
2. **Balances vocabulary size** with sequence length through learned merge rules
3. **Handles any Unicode text** via byte-to-unicode mapping

The GPT-2 tokenizer first splits text using regex patterns (handling contractions, words, numbers, punctuation), then applies BPE merging to each token independently. The result is a sequence of integer IDs that index into the model's embedding table.

=== Usage ===

Use this principle when:
- Converting user prompts to model input format
- Preprocessing text data for language model inference
- Understanding how language models represent text internally
- Implementing custom tokenizers that need OOV-free behavior

Tokenization must occur after model loading (to have the encoder) and before the forward pass (which expects integer IDs).

== Theoretical Basis ==

**Byte Pair Encoding (BPE) Algorithm:**

BPE iteratively merges the most frequent pair of adjacent symbols:

<syntaxhighlight lang="python">
# Abstract BPE algorithm
def bpe(token, merge_ranks):
    word = list(token)  # Start with characters
    while len(word) > 1:
        pairs = get_adjacent_pairs(word)
        best_pair = min(pairs, key=lambda p: merge_ranks.get(p, inf))
        if best_pair not in merge_ranks:
            break  # No more merges possible
        word = merge_pair(word, best_pair)
    return word
</syntaxhighlight>

**GPT-2 BPE Specifics:**

1. **Byte-level vocabulary:** All 256 bytes are mapped to printable Unicode characters to avoid whitespace/control character issues
2. **Regex pre-tokenization:** Text is split into chunks before BPE:
   ```regex
   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
   ```
3. **Learned merges:** The vocabulary is built by learning ~50,000 merge operations from training data

**Token ID Mapping:**

```
"Hello world" → ["Hello", " world"]  (regex split)
             → ["Hello", "Ġworld"]   (byte encoding, Ġ = space)
             → [15496, 995]          (vocabulary lookup)
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Jaymody_PicoGPT_Encoder_Encode]]
