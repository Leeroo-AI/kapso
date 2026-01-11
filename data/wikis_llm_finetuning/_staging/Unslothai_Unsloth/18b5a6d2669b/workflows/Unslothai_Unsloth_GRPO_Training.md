# Workflow: GRPO_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|RL Guide|https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide]]
* [[source::Doc|GRPO Documentation|https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide#training-with-grpo]]
|-
! Domains
| [[domain::LLMs]], [[domain::Reinforcement_Learning]], [[domain::GRPO]], [[domain::Reasoning]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==
End-to-end process for training reasoning models using Group Relative Policy Optimization (GRPO) with Unsloth's memory-efficient reinforcement learning pipeline.

=== Description ===
This workflow implements reinforcement learning fine-tuning using GRPO (Group Relative Policy Optimization), a technique that enables training reasoning models to solve complex problems like mathematics, coding, and logical reasoning. GRPO is more memory-efficient than PPO and produces models that can show their reasoning process.

The workflow typically follows a two-stage approach:
1. **Optional SFT Stage**: Pre-train with supervised fine-tuning on reasoning examples
2. **GRPO Stage**: Reinforce correct reasoning patterns using reward functions

Unsloth's GRPO implementation achieves 80% less VRAM usage compared to standard implementations by:
* Using vLLM for fast batch generation during rollouts
* Memory-efficient reward computation
* Optimized gradient accumulation for RL training

This enables training reasoning models on consumer GPUs with as little as 5GB VRAM.

=== Usage ===
Execute this workflow when:
* You want to train a model to solve reasoning tasks (math, coding, logic)
* You need the model to show step-by-step reasoning in its outputs
* You have verifiable reward signals (correct/incorrect answers)
* You want to improve model capabilities beyond what SFT alone achieves

'''Input requirements:'''
* Base model (optionally pre-trained with SFT)
* Dataset with problems and verifiable answers (e.g., GSM8K, MATH)
* Reward functions that evaluate answer correctness and format compliance
* CUDA-capable GPU with 8GB+ VRAM

'''Expected outputs:'''
* Model that produces structured reasoning traces
* Improved accuracy on reasoning benchmarks
* LoRA adapter or merged model for deployment

== Execution Steps ==

=== Step 1: Model Loading with vLLM ===
[[step::Principle:Unslothai_Unsloth_RL_Model_Loading]]

Initialize the language model with vLLM fast inference enabled using `FastLanguageModel.from_pretrained()` with `fast_inference=True`. This enables efficient batch generation required for GRPO rollouts.

'''Key considerations:'''
* Set `fast_inference=True` to enable vLLM backend for generation
* Configure `gpu_memory_utilization` to balance training and inference memory
* Set `max_lora_rank` to accommodate the LoRA configuration
* Use `load_in_4bit=False` for 16-bit LoRA (recommended for RL)

=== Step 2: RL LoRA Configuration ===
[[step::Principle:Unslothai_Unsloth_RL_LoRA_Configuration]]

Configure Low-Rank Adapters for RL training. GRPO requires trainable parameters for policy updates. Use higher ranks for complex reasoning tasks.

'''Key considerations:'''
* Use rank 64-128 for reasoning tasks (higher than typical SFT)
* Enable `use_gradient_checkpointing="unsloth"` for memory efficiency
* Target all linear layers for comprehensive adaptation
* Consider the memory trade-off between rank and batch size

=== Step 3: Chat Template Configuration ===
[[step::Principle:Unslothai_Unsloth_Chat_Template_Configuration]]

Configure chat templates to establish consistent formatting for reasoning traces. Define special tokens for reasoning boundaries (e.g., `<reasoning>`, `<answer>`).

'''Key considerations:'''
* Use `get_chat_template()` to apply model-specific formatting
* Define clear boundary tokens for reasoning and answer sections
* Ensure the model can generate properly formatted outputs
* Consider using system prompts to guide reasoning behavior

=== Step 4: Dataset Preparation ===
[[step::Principle:Unslothai_Unsloth_RL_Dataset_Preparation]]

Prepare the training dataset with prompts that include problems and expected answer formats. Each example needs a verifiable "answer" field for reward computation.

'''Key considerations:'''
* Format prompts with clear problem statements
* Include expected answer format in system prompt or instructions
* Ensure answers are extractable for reward computation
* Split data appropriately for multi-stage training

=== Step 5: Reward Function Definition ===
[[step::Principle:Unslothai_Unsloth_Reward_Functions]]

Define reward functions that evaluate model completions. GRPO supports multiple reward functions that are combined during training.

'''Typical reward functions:'''
* **Format compliance**: Check if output follows expected structure (reasoning tags, answer tags)
* **Answer correctness**: Verify extracted answer matches ground truth
* **Numerical accuracy**: Score based on how close numerical answers are

'''Key considerations:'''
* Reward functions receive completions in message format
* Return list of scores corresponding to each completion
* Balance positive and negative rewards for stable training
* Consider partial credit for approximately correct answers

=== Step 6: Optional SFT Pre-training ===
[[step::Principle:Unslothai_Unsloth_SFT_Pretraining]]

Optionally pre-train the model with supervised fine-tuning on reasoning examples before GRPO. This warm-starts the policy to produce well-formatted outputs.

'''Key considerations:'''
* Use high-quality reasoning examples (e.g., LIMO dataset)
* Train until format compliance is reasonable
* Use `train_on_responses_only()` to focus on reasoning outputs
* Save checkpoint before GRPO stage

=== Step 7: GRPO Training Configuration ===
[[step::Principle:Unslothai_Unsloth_GRPO_Configuration]]

Configure the GRPO trainer with appropriate hyperparameters for policy optimization. Set generation parameters and training schedule.

'''Key considerations:'''
* Set `num_generations` for rollout batch size (8-16 typical)
* Configure `max_prompt_length` and `max_completion_length`
* Use lower learning rates than SFT (5e-6 typical)
* Set `max_grad_norm` for gradient clipping (0.1 recommended)

=== Step 8: GRPO Training Execution ===
[[step::Principle:Unslothai_Unsloth_GRPO_Execution]]

Execute the GRPO training loop using TRL's GRPOTrainer. The trainer handles rollout generation, reward computation, and policy updates.

'''Training loop:'''
1. Generate multiple completions per prompt using vLLM
2. Compute rewards for each completion using defined functions
3. Calculate relative advantages within each group
4. Update policy using gradient descent on clipped objective

'''Key considerations:'''
* Monitor reward distribution for training stability
* Check format compliance rates during training
* Save checkpoints periodically
* Evaluate on held-out reasoning benchmarks

=== Step 9: Model Saving and Evaluation ===
[[step::Principle:Unslothai_Unsloth_Model_Saving]]

Save the trained model and evaluate on reasoning benchmarks. Compare performance against base model and SFT-only baseline.

'''Key considerations:'''
* Save intermediate checkpoints during training
* Merge to 16-bit for final deployment
* Evaluate on AIME, GSM8K, or task-specific benchmarks
* Test reasoning quality on novel problems

== Execution Diagram ==
{{#mermaid:graph TD
    A[Model Loading with vLLM] --> B[LoRA Configuration]
    B --> C[Chat Template Configuration]
    C --> D[Dataset Preparation]
    D --> E[Reward Function Definition]
    E --> F{SFT Pre-training?}
    F -->|Yes| G[SFT Pre-training]
    F -->|No| H[GRPO Configuration]
    G --> H
    H --> I[GRPO Training Execution]
    I --> J[Model Saving]
    J --> K[Evaluation]
}}

== Related Pages ==
* [[step::Principle:Unslothai_Unsloth_RL_Model_Loading]]
* [[step::Principle:Unslothai_Unsloth_RL_LoRA_Configuration]]
* [[step::Principle:Unslothai_Unsloth_Chat_Template_Configuration]]
* [[step::Principle:Unslothai_Unsloth_RL_Dataset_Preparation]]
* [[step::Principle:Unslothai_Unsloth_Reward_Functions]]
* [[step::Principle:Unslothai_Unsloth_SFT_Pretraining]]
* [[step::Principle:Unslothai_Unsloth_GRPO_Configuration]]
* [[step::Principle:Unslothai_Unsloth_GRPO_Execution]]
* [[step::Principle:Unslothai_Unsloth_Model_Saving]]
