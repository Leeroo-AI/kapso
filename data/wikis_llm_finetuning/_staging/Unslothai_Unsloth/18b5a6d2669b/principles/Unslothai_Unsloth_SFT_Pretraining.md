# Principle: SFT_Pretraining

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|InstructGPT|https://arxiv.org/abs/2203.02155]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Reinforcement_Learning]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Technique for initializing RL policy networks with supervised fine-tuning to establish baseline generation quality before reinforcement learning optimization.

=== Description ===

SFT Pretraining (or warm-up) bridges pre-trained models and RL training. Starting RL from a raw pre-trained model often fails because:

1. **Random policy**: Model generates incoherent text, no reward signal
2. **Exploration inefficiency**: Most generations are garbage
3. **Gradient noise**: High variance updates from poor samples

A brief SFT phase teaches the model basic task structure, providing a competent starting point for RL refinement.

=== Usage ===

Apply SFT Pretraining when:
* Starting GRPO/PPO training from a general model
* The task has complex output format (reasoning, code)
* Initial generations are too poor for reward evaluation

Typically 100-1000 SFT steps are sufficient for warm-up.

== Theoretical Basis ==

=== Policy Initialization ===

RLHF pipeline:

<math>
\text{Pretrained} \xrightarrow{\text{SFT}} \text{SFT Model} \xrightarrow{\text{RL}} \text{Final Model}
</math>

SFT provides a good initialization π_SFT that:
* Generates grammatical, coherent text
* Follows basic task instructions
* Produces parseable outputs for reward functions

=== Response-Only Loss ===

SFT loss is computed only on response tokens:

<math>
\mathcal{L}_{SFT} = -\sum_{t \in \text{response}} \log P_\theta(y_t | x, y_{<t})
</math>

This prevents the model from learning to generate user turns, focusing capacity on response quality.

=== KL Divergence Reference ===

In RL, the SFT model often serves as the KL reference:

<math>
\mathcal{L}_{RL} = -r(y) + \beta \cdot D_{KL}(\pi_\theta || \pi_{SFT})
</math>

This regularizes the RL policy to not drift too far from coherent generation.

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# SFT pretraining for RL (abstract)
def pretrain_for_rl(model, sft_dataset, target_steps=100):
    # 1. Create SFT trainer with response-only loss
    trainer = SFTTrainer(model, sft_dataset)
    trainer = train_on_responses_only(trainer, user_marker, assistant_marker)

    # 2. Brief training to establish baseline
    trainer.train(max_steps=target_steps)

    # 3. Model now generates coherent task-relevant output
    # Ready for RL training

    return model
</syntaxhighlight>

=== When to Skip SFT Pretraining ===

May skip if:
* Model is already instruction-tuned for the task
* Task is simple (single-word answers)
* Pre-trained model already generates reasonable outputs

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_train_on_responses_only]]

