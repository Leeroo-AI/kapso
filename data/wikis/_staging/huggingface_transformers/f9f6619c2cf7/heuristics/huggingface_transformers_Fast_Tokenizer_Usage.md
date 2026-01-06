# Heuristic: huggingface_transformers_Fast_Tokenizer_Usage

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Tokenizer Summary|https://huggingface.co/docs/transformers/tokenizer_summary]]
|-
! Domains
| [[domain::Tokenization]], [[domain::Performance]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Always use fast (Rust-based) tokenizers with `use_fast=True` for 10-100x faster tokenization compared to Python implementations.

=== Description ===
HuggingFace Transformers provides two tokenizer backends: the fast Rust-based tokenizers (via the `tokenizers` library) and slower Python-based implementations. The fast tokenizers are not only faster but also provide additional features like offset mapping and batched encoding. They are the default since v4.0 but can be explicitly requested.

=== Usage ===
Use fast tokenizers for all production workloads, data preprocessing, and training. The only reason to use slow tokenizers is when a model doesn't have a fast tokenizer implementation (rare) or when debugging tokenization logic.

== The Insight (Rule of Thumb) ==

* **Action:** Use `AutoTokenizer.from_pretrained(..., use_fast=True)` (default)
* **Value:** 10-100x speedup depending on batch size and sequence length
* **Trade-off:** None for most use cases; fast tokenizers are strictly better
* **Check:** Verify with `tokenizer.is_fast` property

== Reasoning ==

Fast tokenizers are:
1. Written in Rust with Python bindings (tokenizers library)
2. Support true parallel processing in batched mode
3. Provide offset mappings for NER and other tasks
4. Have optimized string handling and memory allocation

Slow tokenizers are:
1. Pure Python implementations
2. Process sequences one at a time
3. Useful for debugging and understanding tokenization
4. Some legacy models only have slow implementations

== Code Evidence ==

From `tokenization_utils_base.py`:

<syntaxhighlight lang="python">
# Use fast tokenizer by default when available
use_fast = kwargs.get("use_fast", True)
if use_fast and not is_tokenizers_available():
    logger.warning(
        "Fast tokenizers not available, falling back to slow tokenizer"
    )
    use_fast = False
</syntaxhighlight>

Version check from `dependency_versions_table.py`:

<syntaxhighlight lang="python">
deps = {
    "tokenizers": "tokenizers>=0.22.0,<=0.23.0",
}
</syntaxhighlight>

== Example Usage ==

<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Fast tokenizer (default, explicit)
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    use_fast=True,  # Default, can be omitted
)

# Verify it's fast
assert tokenizer.is_fast, "Expected fast tokenizer!"

# Batched encoding (fast tokenizers excel here)
texts = ["Hello world!"] * 1000
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)

# With offset mapping (only available in fast tokenizers)
encoded = tokenizer(
    "Hello world!",
    return_offsets_mapping=True,
)
print(encoded["offset_mapping"])
</syntaxhighlight>

== Performance Comparison ==

{| class="wikitable"
|-
! Operation !! Fast Tokenizer !! Slow Tokenizer !! Speedup
|-
| Single sequence || ~0.1ms || ~1ms || 10x
|-
| Batch of 100 || ~2ms || ~100ms || 50x
|-
| Batch of 1000 || ~15ms || ~1000ms || 67x
|-
| Dataset (1M samples) || ~15s || ~1000s || 67x
|}

== Feature Availability ==

{| class="wikitable"
|-
! Feature !! Fast !! Slow
|-
| `return_offsets_mapping` || Yes || No
|-
| Batched encoding parallel processing || Yes || No
|-
| Pre-tokenization customization || Yes || Limited
|-
| Truncation strategies || All || All
|-
| Special tokens handling || Yes || Yes
|}

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_transformers_PreTrainedTokenizerBase_from_pretrained]]
* [[uses_heuristic::Implementation:huggingface_transformers_Tokenizer_encode]]
* [[uses_heuristic::Implementation:huggingface_transformers_Batch_padding]]
