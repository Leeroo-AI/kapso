# Principle: KV_Cache

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers cache explanation|https://huggingface.co/docs/transformers/kv_cache]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Stored key and value tensors of the tokens already processed, reused at every decoding step so earlier positions are not recomputed.

=== Description ===

Autoregressive generation appends one token at a time. Attention for the new token needs the keys and values of all previous tokens; since those never change, they are computed once and appended to a cache. Per-step cost drops from quadratic to linear in sequence length, while the cache itself grows with length, batch size and number of layers.

=== Usage ===

It is always used for generation; its size is what limits the number of concurrent sequences on a device.

== Theoretical Basis ==

Per token the cache holds 2 * layers * kv_heads * head_dim values. For a 7B-class model with 32 layers, 32 heads of dimension 128 at 16-bit precision, that is about 0.5 MB per token, so a single 4096-token sequence occupies roughly 2 GB.

== Related Pages ==

* [[implemented_by::Implementation:huggingface_transformers_Generate_Use_Cache]]
