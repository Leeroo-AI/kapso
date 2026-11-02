# Principle: Mixed_Precision_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Mixed Precision Training|https://arxiv.org/abs/1710.03740]]
* [[source::Doc|PyTorch Automatic Mixed Precision|https://pytorch.org/docs/stable/amp.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Training scheme that performs most arithmetic in 16-bit floating point while holding 32-bit master copies of the weights, saving memory and raising throughput on tensor cores.

=== Description ===

Half-width formats store each value in two bytes, halving the memory of intermediate tensors and doubling effective memory bandwidth, and modern accelerators multiply them far faster than 32-bit values. The optimizer still updates a 32-bit master copy of the weights, so small increments are not lost to rounding, and numerically delicate operations such as reductions stay in 32 bits.

=== Usage ===

Use it for nearly any training run on hardware with 16-bit tensor-core support; the main decision is which 16-bit format to use and whether the run needs help avoiding underflow.

== Theoretical Basis ==

Two 16-bit formats are in use. IEEE half precision has 5 exponent bits and 10 mantissa bits, so its smallest values underflow easily and its range is narrow. Brain floating point keeps the 8 exponent bits of 32-bit floats and has only 7 mantissa bits, trading resolution for range. Weight updates are accumulated in 32 bits because an increment smaller than about 2^-11 of a half-precision weight would otherwise vanish.

== Related Pages ==

* [[implemented_by::Implementation:pytorch_pytorch_Autocast_GradScaler]]
* [[uses_heuristic::Heuristic:pytorch_pytorch_Gradient_Accumulation_For_Small_Batches]]
