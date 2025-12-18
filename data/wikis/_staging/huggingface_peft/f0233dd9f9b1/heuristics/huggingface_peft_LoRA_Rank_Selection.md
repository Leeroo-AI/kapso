# Heuristic: huggingface_peft_LoRA_Rank_Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|LoRA Paper|https://huggingface.co/papers/2106.09685]]
* [[source::Paper|RSLoRA Paper|https://huggingface.co/papers/2312.03732]]
|-
! Domains
| [[domain::LLMs]], [[domain::Optimization]], [[domain::LoRA]]
|-
! Last Updated
| [[last_updated::2024-12-18 00:00 GMT]]
|}

== Overview ==
Guidelines for selecting LoRA rank (`r`) based on task complexity and memory constraints.

=== Description ===
The LoRA rank (`r`) determines the dimensionality of the low-rank decomposition matrices. A higher rank allows for more expressive adaptations but increases trainable parameters and memory usage. Selecting the right rank is crucial for balancing model quality and computational efficiency.

=== Usage ===
Use this heuristic when configuring `LoraConfig` to determine the optimal `r` value. Consider this heuristic when:
- Starting a new LoRA fine-tuning project
- Memory is constrained and you need to minimize trainable parameters
- Task complexity varies (simple classification vs complex reasoning)

== The Insight (Rule of Thumb) ==
* **Action:** Set LoRA rank `r` in `LoraConfig(r=...)`
* **Value Recommendations:**
  * **Simple tasks (classification, NER):** `r=4-8`
  * **Medium tasks (QA, summarization):** `r=8-16`
  * **Complex tasks (instruction-following, reasoning):** `r=16-64`
  * **Maximum expressiveness:** `r=64-128` (diminishing returns)
* **Default:** `r=8` is a reasonable starting point for most tasks
* **Trade-off:** Higher `r` = more parameters = more memory = potentially better quality

* **Scaling with RSLoRA:**
  * When `use_rslora=True`, scaling factor becomes `lora_alpha/sqrt(r)` instead of `lora_alpha/r`
  * This stabilizes training at higher ranks
  * Recommended when using `r >= 32`

* **Pattern-based ranks:**
  * Use `rank_pattern` to set different ranks for different layers
  * Attention layers often benefit from higher ranks than MLP layers

== Reasoning ==

### Theoretical Foundation
LoRA approximates weight updates as `ΔW = BA` where `B ∈ R^{d×r}` and `A ∈ R^{r×k}`. The rank `r` controls:

1. **Expressiveness**: Higher `r` can capture more complex adaptations
2. **Parameter count**: Trainable params = `r × (d + k)` per adapted layer
3. **Memory**: Both forward/backward pass memory scale with `r`

### Empirical Evidence
The original LoRA paper showed that `r=4-8` was sufficient for many NLU tasks. However, for more complex tasks like instruction-following, practitioners have found that `r=16-64` yields better results.

### RSLoRA Benefit
Rank-Stabilized LoRA (`use_rslora=True`) adjusts the scaling factor to `lora_alpha/sqrt(r)`, which:
- Prevents gradient explosion at high ranks
- Allows effective use of larger rank values
- Maintains stable training dynamics across rank choices

== Code Evidence ==

Default rank value from `config.py:459`:
<syntaxhighlight lang="python">
r: int = field(default=8, metadata={"help": "Lora attention dimension"})
</syntaxhighlight>

RSLoRA documentation from `config.py:488-498`:
<syntaxhighlight lang="python">
use_rslora: bool = field(
    default=False,
    metadata={
        "help": (
            "When set to True, uses [Rank-Stabilized LoRA](https://huggingface.co/papers/2312.03732)"
            " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
            " was proven to work better. Otherwise, it will use the original default"
            " value of `lora_alpha/r`."
        )
    },
)
</syntaxhighlight>

Rank pattern support from `config.py:548-556`:
<syntaxhighlight lang="python">
rank_pattern: Optional[dict] = field(
    default_factory=dict,
    metadata={
        "help": (
            "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
            "For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`."
        )
    },
)
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_peft_LoraConfig_init]]
* [[uses_heuristic::Principle:huggingface_peft_LoRA_Configuration]]
* [[uses_heuristic::Workflow:huggingface_peft_LoRA_Fine_Tuning]]
* [[uses_heuristic::Workflow:huggingface_peft_QLoRA_Training]]
