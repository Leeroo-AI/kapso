# Principle: BPE_Tokenization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Neural Machine Translation of Rare Words with Subword Units|https://arxiv.org/abs/1508.07909]]
* [[source::Paper|Language Models are Unsupervised Multitask Learners|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
* [[source::Blog|Byte Pair Encoding|https://huggingface.co/learn/nlp-course/chapter6/5]]
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::Text_Processing]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Data compression algorithm applied to text that iteratively merges the most frequent character pairs to create a subword vocabulary.

=== Description ===

Byte Pair Encoding (BPE) is a subword tokenization algorithm that bridges the gap between character-level and word-level tokenization. Originally a data compression technique, it was adapted for NLP to handle the open-vocabulary problem where traditional word-level tokenizers fail on rare or unseen words.

The algorithm works in two phases:

1. **Training Phase** (done once, offline):
   - Start with a vocabulary of individual characters
   - Count all adjacent character pairs in the corpus
   - Merge the most frequent pair into a new symbol
   - Repeat until vocabulary reaches desired size

2. **Encoding Phase** (done at inference time):
   - Given a word, repeatedly find and apply the highest-priority merge from the learned merge rules
   - Continue until no more merges can be applied
   - Return the resulting subword sequence

GPT-2 uses a byte-level variant where the base vocabulary consists of 256 byte values, ensuring any text can be tokenized without unknown tokens.

=== Usage ===

Use BPE tokenization when:
- Building or using language models that need to handle any input text
- You want a balance between vocabulary size and sequence length
- Working with multilingual text or text with rare words/typos
- The model was trained with BPE (like GPT-2, GPT-3, etc.)

BPE is preferred over word-level tokenization because it:
- Has no out-of-vocabulary (OOV) tokens
- Handles morphologically rich languages well
- Maintains reasonable sequence lengths (unlike character-level)

== Theoretical Basis ==

The BPE merge algorithm:

<math>
\text{Given merge rules } M = [(p_1, r_1), (p_2, r_2), ...]
</math>

Where each <math>(p_i, r_i)</math> is a pair <math>p_i</math> that merges into replacement <math>r_i</math>, ordered by priority.

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# BPE encoding algorithm
def bpe_encode(word, merge_rules):
    """
    word: string to tokenize
    merge_rules: dict mapping (char1, char2) -> priority
    """
    symbols = list(word)  # Start with characters

    while len(symbols) > 1:
        # Find all adjacent pairs
        pairs = get_adjacent_pairs(symbols)

        # Find the highest priority pair that exists in merge rules
        best_pair = min(pairs, key=lambda p: merge_rules.get(p, infinity))

        if best_pair not in merge_rules:
            break  # No more merges possible

        # Merge all occurrences of best_pair
        symbols = merge_pair(symbols, best_pair)

    return symbols
</syntaxhighlight>

Key properties:
- **Deterministic** - Same input always produces same output
- **Greedy** - Always applies highest-priority available merge
- **Reversible** - Subwords can be concatenated to recover original text

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Jaymody_PicoGPT_Encoder]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_BPE_Caching_LRU]]
