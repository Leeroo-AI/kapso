# Workflow: GRPO_Reinforcement_Learning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL GRPO Documentation|https://huggingface.co/docs/trl/grpo_trainer]]
* [[source::Paper|GRPO Paper|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::LLMs]], [[domain::Reinforcement_Learning]], [[domain::Reasoning]]
|-
! Last Updated
| [[last_updated::2026-01-12 19:00 GMT]]
|}

== Overview ==

Two-stage training pipeline combining supervised fine-tuning with Group Relative Policy Optimization (GRPO) for training reasoning-capable language models.

=== Description ===

This workflow implements a two-stage approach to train language models for mathematical reasoning and structured problem-solving. Stage 1 uses supervised fine-tuning (SFT) on reasoning traces to establish baseline capability. Stage 2 applies GRPO (Group Relative Policy Optimization) to further improve reasoning quality through reward-based optimization.

GRPO is a reinforcement learning technique that samples multiple completions per prompt, scores them with reward functions, and uses the relative rankings to update the policy without requiring a separate critic model.

The workflow covers:
* Initial SFT on reasoning datasets (e.g., LIMO, GSM8K)
* Reward function design for format compliance and answer correctness
* GRPO training with multiple reward signals
* Model merging and validation

=== Usage ===

Execute this workflow when you want to improve a model's reasoning capabilities beyond what supervised fine-tuning alone can achieve. Ideal for:
* Mathematical reasoning (GSM8K, MATH benchmarks)
* Chain-of-thought reasoning tasks
* Tasks where format compliance matters (structured outputs)
* Scenarios where you can define clear reward signals for answer quality

== Execution Steps ==

=== Step 1: Base_Model_Loading ===

Load the base language model with vLLM fast inference enabled for efficient sampling during RL training. Configure quantization and memory settings appropriate for the model size.

'''Key considerations:'''
* Enable `fast_inference=True` for vLLM-backed generation (significantly faster sampling)
* Set appropriate `gpu_memory_utilization` for your hardware
* Use higher `max_lora_rank` if planning for complex adaptations
* Model must support generation for RL sampling loop

=== Step 2: Dataset_Preparation ===

Prepare training datasets for both SFT and GRPO stages. Format reasoning tasks with clear structure including reasoning traces and final answers.

'''What happens:'''
* Convert raw math/reasoning problems to prompt format
* Structure responses with reasoning tags (e.g., `<reasoning>...</reasoning>`)
* Structure answers with solution tags (e.g., `<answer>...</answer>`)
* Split data appropriately for SFT warmup vs GRPO training

'''Key considerations:'''
* Use consistent tagging scheme across all examples
* GRPO dataset needs extractable ground truth answers for reward computation
* Consider dataset quality - LIMO provides high-quality reasoning traces

=== Step 3: LoRA_Adapter_Setup ===

Inject LoRA adapters and configure for training. Target all relevant projection layers for comprehensive adaptation.

'''What happens:'''
* Apply LoRA to attention projections (q, k, v, o) and MLP layers
* Configure adapter rank and alpha scaling
* Enable gradient checkpointing for long sequence handling
* Mark adapter parameters as trainable

=== Step 4: SFT_Warmup_Stage ===

Perform initial supervised fine-tuning to establish baseline reasoning capability before RL optimization. This stage teaches the model the expected output format and basic reasoning patterns.

'''What happens:'''
* Train on high-quality reasoning traces (e.g., LIMO dataset)
* Model learns format conventions (reasoning/answer tags)
* Apply chat template formatting with `train_on_responses_only`
* Creates foundation for subsequent GRPO optimization

'''Key parameters:'''
* Lower learning rate than standard SFT (2e-4 typical)
* Single epoch usually sufficient for warmup
* Use `DataCollatorForSeq2Seq` for proper padding

=== Step 5: Reward_Function_Design ===

Define reward functions that will guide GRPO optimization. Multiple reward signals can be combined for comprehensive feedback.

'''Reward function types:'''
* Format compliance (exact match to expected structure)
* Format approximation (partial credit for near-correct format)
* Answer correctness (comparing extracted answer to ground truth)
* Numerical accuracy (handling floating point comparisons)

'''What happens:'''
* Each reward function receives completions and returns scores
* Functions extract answers from model outputs using regex
* Scores can be positive (reward) or negative (penalty)
* Multiple rewards are combined during GRPO training

=== Step 6: GRPO_Training ===

Execute Group Relative Policy Optimization using the trained model and reward functions. GRPO samples multiple completions per prompt and uses relative rankings to update the policy.

'''What happens:'''
* For each prompt, generate `num_generations` candidate completions
* Score each completion with all reward functions
* Compute advantages based on relative reward rankings within group
* Update policy to increase probability of higher-reward completions

'''Key parameters:'''
* `num_generations` - samples per prompt (8 typical)
* `max_completion_length` - maximum tokens for reasoning
* Learning rate (5e-6, lower than SFT)
* `max_grad_norm` for gradient clipping (0.1 typical)

=== Step 7: Model_Merging_and_Validation ===

Merge trained adapters into base model and validate reasoning performance on held-out test sets.

'''What happens:'''
* Save intermediate checkpoints during training
* Merge LoRA weights into base model for deployment
* Evaluate on benchmarks (AIME, GSM8K) to measure improvement
* Compare performance across different quantization levels

'''Validation considerations:'''
* Test format compliance rate
* Measure exact match accuracy
* Compare base vs SFT vs GRPO performance

== Execution Diagram ==

{{#mermaid:graph TD
    A[Base_Model_Loading] --> B[Dataset_Preparation]
    B --> C[LoRA_Adapter_Setup]
    C --> D[SFT_Warmup_Stage]
    D --> E[Reward_Function_Design]
    E --> F[GRPO_Training]
    F --> G[Model_Merging_and_Validation]
}}

== GitHub URL ==

The executable implementation will be available at:

[[github_url::PENDING_REPO_BUILD]]

<!-- This URL will be populated by the repo builder phase -->
