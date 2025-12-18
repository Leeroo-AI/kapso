# Heuristic: huggingface_peft_Target_Module_Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|LoRA Paper|https://huggingface.co/papers/2106.09685]]
|-
! Domains
| [[domain::LLMs]], [[domain::Optimization]], [[domain::LoRA]]
|-
! Last Updated
| [[last_updated::2024-12-18 00:00 GMT]]
|}

== Overview ==
Guidelines for selecting which modules to apply LoRA adapters using `target_modules` configuration.

=== Description ===
The `target_modules` parameter in `LoraConfig` determines which layers receive LoRA adapters. Choosing the right target modules is crucial for balancing adaptation quality with parameter efficiency. PEFT supports automatic selection, wildcard patterns, and explicit module lists.

=== Usage ===
Use this heuristic when configuring `LoraConfig.target_modules` to decide which layers should be adapted. Consider this when:
- Starting LoRA fine-tuning on a new model architecture
- Optimizing for memory by targeting fewer modules
- Maximizing quality by targeting more modules

== The Insight (Rule of Thumb) ==

### Strategy Options

* **Quick Start (`"all-linear"`):**
  * Targets all linear layers (excluding output layer for PreTrainedModel)
  * Most comprehensive but highest memory usage
  * Good for maximum quality when memory allows
  * Action: `target_modules="all-linear"`

* **Attention-Only (Classic LoRA):**
  * Target only Q and V projections (original LoRA paper recommendation)
  * Good balance of quality and efficiency
  * Action: `target_modules=["q_proj", "v_proj"]`

* **Full Attention:**
  * Target all attention projections (Q, K, V, O)
  * Better than attention-only for most tasks
  * Action: `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`

* **Attention + MLP:**
  * Target attention and MLP layers
  * Best quality, moderate memory
  * Action: `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

### Model-Specific Conventions

* **Llama/Mistral:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* **GPT-2/GPT-J:** `c_attn`, `c_proj`, `c_fc`
* **BERT:** `query`, `key`, `value`, `dense`
* **T5:** `q`, `k`, `v`, `o`, `wi`, `wo`

* **Trade-off:** More modules = more parameters = better quality but higher memory

* **Exclusion:** Use `exclude_modules` to skip specific patterns from `all-linear`

== Reasoning ==

### Original LoRA Findings
The original LoRA paper experimented with adapting different subsets of weight matrices and found that adapting attention matrices (especially Q and V) provided most of the benefit. However, recent practice shows that adapting more layers often improves results.

### Empirical Best Practices
For instruction-following and complex tasks, targeting all linear layers or at least full attention + MLP layers typically yields better results than Q/V only. The `all-linear` shortcut makes this easy.

### Automatic Detection
When `target_modules=None`, PEFT attempts to automatically detect appropriate modules based on model architecture. This works for known architectures but may fail for custom models.

== Code Evidence ==

`all-linear` support from `config.py:460-472`:
<syntaxhighlight lang="python">
target_modules: Optional[Union[list[str], str]] = field(
    default=None,
    metadata={
        "help": (
            "List of module names or regex expression of the module names to replace with LoRA. "
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
            "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
            "(if the model is a PreTrainedModel, the output layer excluded). "
            "If not specified, modules will be chosen according to the model architecture, If the architecture is "
            "not known, an error will be raised -- in this case, you should specify the target modules manually."
        ),
    },
)
</syntaxhighlight>

Module exclusion from `config.py:475-478`:
<syntaxhighlight lang="python">
exclude_modules: Optional[Union[list[str], str]] = field(
    default=None,
    metadata={"help": "List of module names or regex expression of the module names to exclude from Lora."},
)
</syntaxhighlight>

Architecture-specific defaults from model configs in transformers integration:
<syntaxhighlight lang="python">
# Example for Llama (detected automatically)
target_modules = ["q_proj", "v_proj"]  # Minimal default
# OR with all-linear:
target_modules = "all-linear"  # All linear layers
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_peft_LoraConfig_init]]
* [[uses_heuristic::Principle:huggingface_peft_LoRA_Configuration]]
* [[uses_heuristic::Workflow:huggingface_peft_LoRA_Fine_Tuning]]
