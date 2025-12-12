{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Packing: Towards 2x NLP Training Speedup|https://arxiv.org/abs/2107.02027]]
* [[source::Doc|TRL Packing|https://huggingface.co/docs/trl/sft_trainer]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
|-
! Domains
| [[domain::Optimization]], [[domain::Training]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Training throughput optimization by packing multiple short sequences into a single training example to eliminate padding waste.

=== Description ===
Standard training pads all sequences to the maximum length, wasting compute on padding tokens. Sequence packing concatenates multiple short sequences (with separator tokens) into single examples up to max_seq_length, eliminating padding overhead. This can provide up to 2x speedup for datasets with variable-length samples.

=== Usage ===
Use this heuristic when your dataset has **variable-length sequences**, especially if many are shorter than `max_seq_length`. Most effective for instruction-tuning datasets where conversations vary in length.

== The Insight (Rule of Thumb) ==
* **Action:** Enable packing in SFTTrainer configuration.
* **Value:** Set `packing = True` in SFTConfig or SFTTrainer.
* **When to Use:** Dataset avg length << max_seq_length

<syntaxhighlight lang="python">
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    args = SFTConfig(
        packing = True,              # Enable sequence packing
        max_seq_length = 2048,       # Packed sequences up to this length
        # ... other args
    ),
)
</syntaxhighlight>

* **Speedup Estimates:**

{| class="wikitable"
! Avg Sequence Length !! Max Sequence Length !! Approximate Speedup
|-
|| 256 || 2048 || ~5-8x
|-
|| 512 || 2048 || ~3-4x
|-
|| 1024 || 2048 || ~1.5-2x
|-
|| 1500 || 2048 || ~1.2-1.3x
|}

* **Trade-off:**
  * Packing enabled: Faster training, may slightly affect attention patterns at boundaries
  * Packing disabled: Cleaner sequence boundaries, more padding waste

== Reasoning ==
Without packing, a batch with sequences of length [200, 300, 150] padded to 2048:
* Total tokens processed: 3 × 2048 = 6144
* Actual content: 200 + 300 + 150 = 650
* Efficiency: ~10%

With packing, these sequences become one 650-token example:
* Total tokens: 650
* Efficiency: 100%
* Result: 10x less wasted compute

Unsloth's SFTTrainer integrates with TRL's packing implementation, which handles attention masking to prevent cross-sequence attention.

== Related Pages ==
=== Used By ===
* [[uses_heuristic::Workflow:QLoRA_Finetuning]]
* [[uses_heuristic::Implementation:TRL_SFTTrainer]]
* [[uses_heuristic::Principle:Supervised_Fine_Tuning]]

