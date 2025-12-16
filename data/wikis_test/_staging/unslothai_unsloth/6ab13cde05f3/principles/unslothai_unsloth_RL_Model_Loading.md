{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|RL Guide|https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::Model_Loading]], [[domain::vLLM]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of loading models with vLLM backend enabled for efficient batch generation during reinforcement learning training.

=== Description ===

RL model loading requires special configuration:
- vLLM backend for fast generation (`fast_inference=True`)
- GPU memory reservation for gradients (`gpu_memory_utilization < 1.0`)
- Higher LoRA rank support for RL capacity (`max_lora_rank=64`)

== Practical Guide ==

=== Load with vLLM Backend ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
    fast_inference=True,        # Enable vLLM
    gpu_memory_utilization=0.6, # Reserve 40% for gradients
    max_lora_rank=64,           # Support higher LoRA ranks
)
</syntaxhighlight>

=== Memory Allocation ===
<syntaxhighlight lang="python">
# gpu_memory_utilization controls vLLM's memory share
# Lower values leave more memory for PyTorch gradients

# For 24GB GPU:
# - 0.5: 12GB vLLM, 12GB gradients (balanced)
# - 0.6: 14.4GB vLLM, 9.6GB gradients (more generation capacity)
# - 0.7: 16.8GB vLLM, 7.2GB gradients (aggressive)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_PatchFastRL]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
