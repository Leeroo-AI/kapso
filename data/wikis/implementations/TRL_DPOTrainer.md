{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|TRL GitHub|https://github.com/huggingface/trl]]
* [[source::Doc|TRL DPOTrainer|https://huggingface.co/docs/trl/dpo_trainer]]
* [[source::Paper|DPO Paper|https://arxiv.org/abs/2305.18290]]
|-
! Domains
| [[domain::RLHF]], [[domain::Alignment]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Concrete tool for Direct Preference Optimization (DPO) training provided by HuggingFace TRL library, compatible with Unsloth.

=== Description ===
`DPOTrainer` implements the DPO algorithm for aligning language models with human preferences without explicit reward modeling. It takes pairs of chosen/rejected responses and trains the model to prefer chosen responses. Works seamlessly with Unsloth-patched models for 2x faster alignment training.

=== Usage ===
Import this class when aligning a language model using preference data (chosen vs rejected pairs). Use after initial SFT training to further improve response quality. Requires a dataset with `prompt`, `chosen`, and `rejected` columns.

== Code Signature ==
<syntaxhighlight lang="python">
from trl import DPOTrainer, DPOConfig

class DPOTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: Optional[PreTrainedModel] = None,
        args: Optional[DPOConfig] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        **kwargs
    ):
        ...
    
    def train(self) -> TrainOutput:
        """Run DPO training loop."""
        ...
</syntaxhighlight>

== I/O Contract ==
* **Consumes:**
    * `model`: Unsloth-patched model (usually after SFT)
    * `ref_model`: Reference model (optional, uses initial model if None)
    * `train_dataset`: Dataset with `prompt`, `chosen`, `rejected` columns
    * `beta`: Temperature parameter for DPO loss
* **Produces:**
    * Aligned model that prefers chosen responses
    * Training logs and metrics

== Example Usage ==
<syntaxhighlight lang="python">
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# Load preference dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# Configure DPO training
args = DPOConfig(
    output_dir = "dpo_outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-5,
    max_steps = 100,
    beta = 0.1,
    optim = "adamw_8bit",
)

# Create trainer
trainer = DPOTrainer(
    model = model,
    ref_model = None,  # Uses model's initial state
    args = args,
    train_dataset = dataset,
    tokenizer = tokenizer,
)

trainer.train()
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:Unsloth_CUDA_Environment]]
* [[requires_env::Environment:Unsloth_Docker_Environment]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Learning_Rate_Tuning]]
* [[uses_heuristic::Heuristic:Gradient_Accumulation_Strategy]]

