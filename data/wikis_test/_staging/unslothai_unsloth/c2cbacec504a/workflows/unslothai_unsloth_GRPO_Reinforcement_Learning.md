{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Doc|RL Guide|https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide]]
* [[source::Blog|GRPO Blog|https://unsloth.ai/blog/grpo]]
|-
! Domains
| [[domain::LLMs]], [[domain::Reinforcement_Learning]], [[domain::GRPO]], [[domain::Reasoning]]
|-
! Last Updated
| [[last_updated::2025-12-17 14:00 GMT]]
|}

== Overview ==
End-to-end process for training reasoning-enhanced language models using Group Relative Policy Optimization (GRPO) reinforcement learning, optionally combined with initial supervised fine-tuning warmup.

=== Description ===
This workflow implements a two-stage training pipeline that transforms base LLMs into reasoning models capable of structured problem-solving. The approach combines:

1. **Optional SFT Warmup** (Stage 1): Initial supervised fine-tuning on high-quality reasoning demonstrations (e.g., LIMO dataset) to establish baseline reasoning patterns
2. **GRPO Training** (Stage 2): Reinforcement learning that optimizes model outputs against custom reward functions, encouraging correct formatting, accurate answers, and step-by-step reasoning

GRPO is particularly effective for mathematical reasoning tasks (GSM8K, AIME) where format compliance and answer correctness can be programmatically verified. Unsloth provides optimized RL implementations using 80% less VRAM than standard approaches through shared base model references and efficient gradient masking.

Key capabilities:
* Custom reward function support for multi-criteria optimization
* vLLM-accelerated inference for fast completion generation during RL
* Memory-efficient training with shared reference models
* Support for GRPO, GSPO, DPO, ORPO, and PPO variants

=== Usage ===
Execute this workflow when:
* You want to improve a model's reasoning capabilities on structured tasks
* You have programmatically verifiable success criteria (format compliance, answer correctness)
* You need to train custom reward models or use rule-based reward functions
* You want to transform instruction-following models into reasoning models

'''Input requirements:'''
* Base model (can be pre-fine-tuned with LoRA)
* Training dataset with prompts and ground truth answers
* Custom reward functions defining optimization objectives
* Sufficient VRAM for generation + training (typically 16-48GB)

'''Output:'''
* LoRA adapter weights optimized for reasoning tasks
* Optionally: merged model for deployment

== Execution Steps ==

=== Step 1: Model Loading with vLLM ===
[[step::Principle:unslothai_unsloth_RL_Model_Loading]]

Load the base model with vLLM fast inference enabled. For RL training, the model needs to generate multiple completions per prompt efficiently, which vLLM's batched inference accelerates significantly. The loader configures both the training model and the implicit reference model.

'''What happens:'''
* Base model loaded with 4-bit or 16-bit quantization
* vLLM inference engine initialized for fast generation
* GPU memory utilization configured for generation + training workloads
* Max LoRA rank pre-allocated for dynamic adapter loading

'''Key parameters:'''
* `fast_inference`: True (enables vLLM backend)
* `gpu_memory_utilization`: Fraction of VRAM for vLLM (typically 0.5-0.8)
* `max_lora_rank`: Pre-allocate for LoRA adapters

=== Step 2: LoRA Adapter Setup ===
[[step::Principle:unslothai_unsloth_LoRA_Configuration]]

Apply LoRA adapters to the model using the same approach as standard fine-tuning. For RL, larger ranks (64+) are often beneficial as the model needs capacity to learn new reasoning patterns.

'''Key considerations:'''
* Higher LoRA rank typically improves RL training quality
* Use `use_gradient_checkpointing = "unsloth"` for long-context training
* Target all linear layers for maximum adaptability

=== Step 3: Chat Template Configuration ===
[[step::Principle:unslothai_unsloth_Chat_Template_Setup]]

Configure the tokenizer with the appropriate chat template for the model family. Consistent formatting is critical for RL since reward functions often check for specific format markers (e.g., `<reasoning>...</reasoning><answer>...</answer>`).

'''What happens:'''
* Chat template applied matching model family (llama-3.1, chatML, etc.)
* Special tokens configured for generation prompts
* Template includes system prompt defining expected output format

=== Step 4: SFT Warmup (Optional) ===
[[step::Principle:unslothai_unsloth_SFT_Training]]

Perform optional supervised fine-tuning on high-quality reasoning demonstrations. This warmup stage helps the model learn the expected output format and basic reasoning patterns before RL optimization.

'''What happens:'''
* Model trained on reasoning demonstrations (e.g., LIMO, math solutions)
* `train_on_responses_only()` masks non-response tokens from loss
* Establishes baseline format compliance before RL

'''Key considerations:'''
* Warmup typically 1 epoch on demonstration data
* Saves checkpoint for RL stage to continue from
* Can skip if model already follows expected format

=== Step 5: Reward Function Definition ===
[[step::Principle:unslothai_unsloth_Reward_Function_Interface]]

Define custom reward functions that evaluate model completions. Reward functions receive batches of prompts and completions, returning numerical scores that guide optimization. Multiple reward functions can be combined.

'''Reward function signature:'''
```
def reward_func(prompts, completions, **kwargs) -> List[float]:
    # Evaluate each completion and return reward scores
```

'''Common reward types:'''
* Format compliance: Check for required tags/structure
* Answer correctness: Compare extracted answer to ground truth
* Length penalties: Discourage overly verbose outputs
* Fluency rewards: Evaluate reasoning quality

=== Step 6: GRPO Trainer Configuration ===
[[step::Principle:unslothai_unsloth_GRPO_Configuration]]

Configure the GRPOTrainer with generation and training parameters. GRPO generates multiple completions per prompt, computes rewards, and updates the policy based on relative performance within each group.

'''Key parameters:'''
* `num_generations`: Completions per prompt (typically 4-16)
* `max_prompt_length`: Maximum input length
* `max_completion_length`: Maximum generation length
* `learning_rate`: Typically 5e-6 to 2e-5 for RL
* `per_device_train_batch_size`: Often 1-2 due to generation overhead
* `gradient_accumulation_steps`: Increase effective batch size

=== Step 7: GRPO Training Execution ===
[[step::Principle:unslothai_unsloth_GRPO_Training]]

Execute the GRPO training loop. For each batch, the trainer generates multiple completions using vLLM, evaluates them with reward functions, and updates the policy to increase probability of high-reward outputs relative to low-reward ones.

'''What happens:'''
* Prompts sampled from training dataset
* Multiple completions generated per prompt using vLLM
* Each completion scored by all reward functions
* Policy updated based on relative rewards within generation group
* Reference model (frozen) used for KL divergence regularization

=== Step 8: Model Saving and Evaluation ===
[[step::Principle:unslothai_unsloth_Model_Saving]]

Save the trained RL model and optionally merge to full precision. Evaluation should compare performance on held-out reasoning tasks against the base model and SFT checkpoint.

'''Save options:'''
* Save LoRA checkpoint for continued training
* Merge to 16-bit for deployment
* Export to GGUF for local inference

== Execution Diagram ==
{{#mermaid:graph TD
    A[Model Loading + vLLM] --> B[LoRA Setup]
    B --> C[Chat Template Config]
    C --> D{SFT Warmup?}
    D -->|Yes| E[SFT Training]
    D -->|No| F[Reward Definition]
    E --> F
    F --> G[GRPO Config]
    G --> H[GRPO Training]
    H --> I[Save & Evaluate]
}}

== Related Pages ==
* [[step::Principle:unslothai_unsloth_RL_Model_Loading]]
* [[step::Principle:unslothai_unsloth_LoRA_Configuration]]
* [[step::Principle:unslothai_unsloth_Chat_Template_Setup]]
* [[step::Principle:unslothai_unsloth_SFT_Training]]
* [[step::Principle:unslothai_unsloth_Reward_Function_Interface]]
* [[step::Principle:unslothai_unsloth_GRPO_Configuration]]
* [[step::Principle:unslothai_unsloth_GRPO_Training]]
* [[step::Principle:unslothai_unsloth_Model_Saving]]
