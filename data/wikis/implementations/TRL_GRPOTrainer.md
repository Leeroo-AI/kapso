{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|TRL GitHub|https://github.com/huggingface/trl]]
* [[source::Paper|DeepSeekMath GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Doc|Unsloth GRPO Guide|https://docs.unsloth.ai/]]
|-
! Domains
| [[domain::RLHF]], [[domain::Reasoning]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Concrete tool for Group Relative Policy Optimization (GRPO) training provided by HuggingFace TRL library, optimized for Unsloth.

=== Description ===
`GRPOTrainer` implements the GRPO algorithm, a reinforcement learning method that improves reasoning capabilities by training on groups of responses with relative rewards. Popularized by DeepSeek for math reasoning, it generates multiple responses per prompt and uses group-relative rewards for training. Unsloth provides significant speedups for GRPO workflows.

=== Usage ===
Import this class when training models for improved reasoning (math, code, logic). Use with a reward function or model that can score response quality. Particularly effective for enhancing chain-of-thought and step-by-step reasoning capabilities.

== Code Signature ==
<syntaxhighlight lang="python">
from trl import GRPOTrainer, GRPOConfig

class GRPOTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        reward_model: Optional[PreTrainedModel] = None,
        reward_fn: Optional[Callable] = None,
        num_generations: int = 4,
        **kwargs
    ):
        ...
    
    def train(self) -> TrainOutput:
        """Run GRPO training loop."""
        ...
</syntaxhighlight>

== I/O Contract ==
* **Consumes:**
    * `model`: Unsloth-patched model with LoRA adapters
    * `train_dataset`: Dataset with prompts (questions/problems)
    * `reward_model` or `reward_fn`: Scoring mechanism for responses
    * `num_generations`: Number of responses to generate per prompt
* **Produces:**
    * Model with improved reasoning capabilities
    * Training logs with reward metrics

== Example Usage ==
<syntaxhighlight lang="python">
from trl import GRPOTrainer, GRPOConfig

# Define reward function (example for math)
def reward_fn(responses, ground_truths):
    rewards = []
    for resp, gt in zip(responses, ground_truths):
        # Check if answer matches
        rewards.append(1.0 if gt in resp else 0.0)
    return rewards

# Configure GRPO training
args = GRPOConfig(
    output_dir = "grpo_outputs",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate = 1e-5,
    max_steps = 500,
    num_generations = 4,  # Generate 4 responses per prompt
    optim = "adamw_8bit",
)

# Create trainer
trainer = GRPOTrainer(
    model = model,
    args = args,
    train_dataset = dataset,
    tokenizer = tokenizer,
    reward_fn = reward_fn,
)

trainer.train()
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:Unsloth_CUDA_Environment]]
* [[requires_env::Environment:Unsloth_Docker_Environment]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Learning_Rate_Tuning]]
* [[uses_heuristic::Heuristic:Batch_Size_Optimization]]

