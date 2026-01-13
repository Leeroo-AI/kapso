# Principle: LoRA_Adapter_Injection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation of Large Language Models|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Deep_Learning]], [[domain::Parameter_Efficient_Training]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for injecting low-rank adapter matrices into a frozen pre-trained model to enable parameter-efficient fine-tuning.

=== Description ===

LoRA (Low-Rank Adaptation) enables efficient fine-tuning by injecting trainable rank decomposition matrices into frozen pre-trained model layers. Instead of updating all model parameters, LoRA adds small adapter weights to specific layers (typically attention projections and MLP layers), reducing the number of trainable parameters by orders of magnitude.

For a pre-trained weight matrix W ∈ R^(d×k), LoRA adds a low-rank decomposition:
- W' = W + BA where B ∈ R^(d×r) and A ∈ R^(r×k)
- Only A and B are trained, while W remains frozen
- The rank r is typically 8-64, much smaller than d or k

This approach reduces memory requirements, speeds up training, and produces small adapter files that can be easily shared and combined.

=== Usage ===

Use this principle when:
* Fine-tuning large language models with limited compute resources
* Training task-specific adapters that can be swapped at inference time
* Preserving the base model's capabilities while adding new skills
* After loading a quantized base model in a QLoRA workflow

This is the second step in QLoRA fine-tuning, applied after model loading and before data preparation.

== Theoretical Basis ==

The LoRA weight update is formulated as:

<math>
h = W_0 x + \Delta W x = W_0 x + BA x
</math>

Where:
- W_0 is the frozen pre-trained weight
- B ∈ R^(d×r), A ∈ R^(r×k) are the low-rank adapter matrices
- r << min(d, k) is the rank (typically 8-64)
- A is initialized with random Gaussian, B is initialized to zero

'''Key Properties:'''
1. **Initialization**: At training start, BA = 0, so h = W_0 x (no change from pre-trained)
2. **Scaling**: The adapter output is scaled by α/r where α is lora_alpha
3. **Target Modules**: Typically applied to attention (q,k,v,o projections) and MLP (gate, up, down projections)

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Abstract algorithm for LoRA injection
for layer in model.layers:
    for name in target_modules:  # ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        original_weight = layer[name].weight  # Frozen
        A = torch.randn(r, in_features) * init_scale
        B = torch.zeros(out_features, r)
        # New forward: output = original_weight @ x + (B @ A @ x) * (alpha / r)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_get_peft_model]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_LoRA_Rank_Selection]]
