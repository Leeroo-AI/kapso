{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
* [[source::Paper|Training Deep Nets with Sublinear Memory|https://arxiv.org/abs/1604.06174]]
|-
! Domains
| [[domain::Optimization]], [[domain::Memory_Management]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Memory optimization using Unsloth's custom gradient checkpointing implementation to reduce VRAM by 50-60% while maintaining 2x training speed.

=== Description ===
Unsloth provides an enhanced gradient checkpointing implementation that goes beyond standard PyTorch checkpointing. By setting `use_gradient_checkpointing = "unsloth"`, you activate optimizations that reduce VRAM usage by approximately 30% more than standard gradient checkpointing, while still achieving 2x faster training than HuggingFace baselines. This is achieved through intelligent recomputation strategies and kernel fusion optimizations.

=== Usage ===
Use this heuristic when you are **VRAM constrained** (e.g., CUDA OOM errors) or need to fit larger batch sizes on limited hardware. Essential for fine-tuning 7B+ parameter models on consumer GPUs (16-24GB VRAM). Standard practice for QLoRA workflows with Unsloth.

== The Insight (Rule of Thumb) ==
* **Action:** Set `use_gradient_checkpointing = "unsloth"` in `FastLanguageModel.get_peft_model()` or `FastModel.get_peft_model()`.
* **Value:** String `"unsloth"` (not boolean `True`).
* **Trade-off:** Reduces VRAM by ~30% more than `True`, with ~5-10% slower training than no checkpointing (but 2x faster than HF baseline with checkpointing).
* **Compatibility:** Works with all Unsloth-supported models. Requires `use_cache=False` during training.

<syntaxhighlight lang="python">
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # <-- Key setting
    random_state = 3407,
)
</syntaxhighlight>

== Reasoning ==
Deep Transformer models store massive activation tensors during the forward pass for backpropagation. Unsloth's `"unsloth"` mode implements custom CUDA kernels that:
1. Selectively checkpoint only the most memory-intensive layers
2. Fuse operations to reduce intermediate tensor storage
3. Recompute activations efficiently during backward pass

Benchmarks show Llama-3.1 8B training VRAM drops from ~20GB to ~8GB with this setting, enabling fine-tuning on RTX 3060/4060 class GPUs.

== Related Pages ==
=== Used By ===
* [[uses_heuristic::Workflow:QLoRA_Finetuning]]
* [[uses_heuristic::Workflow:GRPO_Reinforcement_Learning]]
* [[uses_heuristic::Workflow:DPO_Alignment]]
* [[uses_heuristic::Implementation:Unsloth_get_peft_model]]
* [[uses_heuristic::Principle:Gradient_Checkpointing]]

