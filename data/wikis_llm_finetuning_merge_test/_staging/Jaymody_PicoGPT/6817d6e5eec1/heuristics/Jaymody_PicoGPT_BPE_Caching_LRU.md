# Heuristic: BPE_Caching_LRU

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Repo|OpenAI GPT-2|https://github.com/openai/gpt-2/blob/master/src/encoder.py]]
|-
! Domains
| [[domain::NLP]], [[domain::Optimization]], [[domain::Tokenization]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Cache BPE tokenization results using LRU cache and instance-level dictionaries to avoid redundant computations.

=== Description ===
Byte Pair Encoding (BPE) tokenization involves iteratively merging character pairs according to learned merge rules. This process is computationally expensive, especially for longer tokens. Caching results at both the function level (`@lru_cache` for `bytes_to_unicode`) and instance level (dictionary cache for `bpe` method) avoids redundant computation when the same tokens or character mappings are requested multiple times.

=== Usage ===
Use this heuristic when implementing **BPE Tokenizers** or any tokenization scheme with expensive per-token computations. Especially valuable when processing text with repeated patterns, common words, or during iterative generation where partial sequences are re-encoded.

== The Insight (Rule of Thumb) ==
* **Action:** Use `@lru_cache()` for pure functions and instance dictionary `self.cache` for method results
* **Value:** Unbounded `lru_cache` for static mappings; dictionary cache for per-Encoder BPE results
* **Trade-off:** Memory usage grows with vocabulary diversity; negligible for standard text
* **Compatibility:** Python 3.2+ for `functools.lru_cache`; dictionary caching works everywhere

== Reasoning ==
BPE tokenization has two expensive operations:

1. **bytes_to_unicode mapping** (static, computed once):
   - Creates a mapping from 256 byte values to unicode characters
   - Same for all text, never changes
   - Perfect candidate for `@lru_cache()` - compute once, reuse forever

2. **BPE merge algorithm** (per-token):
   - For each word, iteratively apply merge rules
   - Complexity grows with word length and merge vocabulary size
   - Common words (articles, prepositions, frequent terms) appear repeatedly
   - Caching in `self.cache` dictionary avoids re-computing for repeated words

Memory vs speed tradeoff:
- For typical English text, ~10,000-50,000 unique words
- Each cache entry is a string (the merged token representation)
- Total cache memory: typically < 1MB
- Speed improvement: 10-100x for repeated tokenization

== Code Evidence ==

From `encoder.py:L7,12-32` (LRU cache for bytes_to_unicode):
<syntaxhighlight lang="python">
from functools import lru_cache

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    ...
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
</syntaxhighlight>

From `encoder.py:L55,60-62,98` (instance cache for BPE):
<syntaxhighlight lang="python">
def __init__(self, encoder, bpe_merges, errors="replace"):
    ...
    self.cache = {}
    ...

def bpe(self, token):
    if token in self.cache:
        return self.cache[token]
    ...
    # After computing the BPE result:
    self.cache[token] = word
    return word
</syntaxhighlight>

The pattern:
1. Check cache at function entry
2. Return cached result if available
3. Compute expensive operation only on cache miss
4. Store result in cache before returning

== Related Pages ==
* [[used_by::Implementation:Jaymody_PicoGPT_Encoder]]
* [[used_by::Principle:Jaymody_PicoGPT_BPE_Tokenization]]
