# Heuristic: Embedding_Learning_Rate_Tip

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|trainer.py|https://github.com/unslothai/unsloth/blob/main/unsloth/trainer.py]]
|-
! Domains
| [[domain::Optimization]], [[domain::Training]], [[domain::Embeddings]]
|-
! Last Updated
| [[last_updated::2026-01-09 12:00 GMT]]
|}

== Overview ==
Use a lower learning rate for embedding layers (5e-5) compared to LoRA parameters (2e-4) for stable training.

=== Description ===
When training with embeddings unfrozen (via `use_exact_model_tokens=True` in `get_peft_model`), embedding layers require a lower learning rate than LoRA adapter parameters. This is because embeddings are pre-trained on large corpora and drastic updates can destabilize the model.

Unsloth provides `embedding_learning_rate` in `UnslothTrainingArguments` to set a separate learning rate for embedding layers.

=== Usage ===
Use this heuristic when:
- **Training with unfrozen embeddings:** Using `use_exact_model_tokens=True`
- **Adding new tokens:** Extending vocabulary for domain-specific terms
- **Observing training instability:** Loss spikes or NaN values

== The Insight (Rule of Thumb) ==
* **Action:** Set `embedding_learning_rate` in `UnslothTrainingArguments`
* **Value:**
  - Embedding LR: `5e-5` (default recommendation)
  - Main LR: `2e-4` (standard for LoRA)
  - Ratio: embedding_lr ≈ 0.25 × main_lr
* **Trade-off:** Lower embedding LR = more stable but slower embedding adaptation

== Reasoning ==
Embedding layers have fundamentally different properties than LoRA adapters:

1. **Pre-trained knowledge:** Embeddings encode word/token semantics learned from massive corpora
2. **High impact:** Small embedding changes affect every layer of the model
3. **Different gradients:** Embedding gradients tend to be larger in magnitude

Using the same learning rate for embeddings and LoRA often causes:
- Loss instability
- Catastrophic forgetting of language understanding
- Embedding drift that degrades generation quality

The 5e-5 default is empirically validated to balance adaptation speed with stability.

== Code Evidence ==

Embedding optimizer from `trainer.py:139-179`:
<syntaxhighlight lang="python">
def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr = 5e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = {
        "non_embeddings": {},
        "embeddings": {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[: -len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".") + 1 :]
            print(
                f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}."
            )
            param_groups["embeddings"][name] = param
        else:
            param_groups["non_embeddings"][name] = param

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["non_embeddings"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embeddings"].values()),
            "weight_decay": weight_decay,
            "lr": embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Unslothai_Unsloth_UnslothTrainingArguments]]
