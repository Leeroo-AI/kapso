{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|8-bit Optimizers|https://arxiv.org/abs/2110.02861]]
* [[source::Repo|bitsandbytes|https://github.com/TimDettmers/bitsandbytes]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
|-
! Domains
| [[domain::Optimization]], [[domain::Memory_Management]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Memory optimization using 8-bit Adam optimizer to reduce optimizer state memory by 75% with negligible quality impact.

=== Description ===
Standard Adam optimizer stores two momentum states per parameter in FP32, consuming significant memory for large models. The 8-bit Adam variant from bitsandbytes quantizes these states to 8-bit integers, reducing optimizer memory by ~75% with minimal impact on training quality. Unsloth integrates seamlessly with 8-bit optimizers.

=== Usage ===
Use this heuristic when **VRAM constrained** or training models where optimizer states consume significant memory. Standard practice for QLoRA fine-tuning with Unsloth.

== The Insight (Rule of Thumb) ==
* **Action:** Set `optim = "adamw_8bit"` in training configuration.
* **Value:** String `"adamw_8bit"`
* **Alternative:** `"paged_adamw_8bit"` for additional memory savings (offloads to CPU when needed)

<syntaxhighlight lang="python">
from trl import SFTConfig

args = SFTConfig(
    optim = "adamw_8bit",           # 8-bit Adam optimizer
    # OR for extreme memory constraints:
    optim = "paged_adamw_8bit",     # With CPU offloading
    # ... other args
)
</syntaxhighlight>

* **Memory Savings:**

{| class="wikitable"
! Optimizer !! Memory per Parameter !! Relative to FP32 Adam
|-
|| adamw (FP32) || 8 bytes || 100%
|-
|| adamw_8bit || 2 bytes || 25%
|-
|| paged_adamw_8bit || 2 bytes + paging || 25% (can offload)
|}

* **Trade-off:** Negligible quality loss (<0.1% in benchmarks), significant memory savings.

== Reasoning ==
Adam optimizer maintains two momentum buffers (m and v) per parameter:
* FP32 model: 8 bytes per parameter for optimizer states
* 7B model: ~56GB just for optimizer states in FP32!

8-bit quantization of these states reduces this to ~14GB while maintaining training dynamics through dynamic quantization. Combined with 4-bit model quantization, this makes training 7B+ models feasible on consumer hardware.

Unsloth's optimizations stack with 8-bit optimizers for maximum memory efficiency.

== Related Pages ==
=== Used By ===
* [[uses_heuristic::Workflow:QLoRA_Finetuning]]
* [[uses_heuristic::Implementation:TRL_SFTTrainer]]
* [[uses_heuristic::Implementation:TRL_SFTConfig]]
* [[uses_heuristic::Implementation:BitsAndBytes_4bit_Quantization]]

