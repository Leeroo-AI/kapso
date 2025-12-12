{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|TRL GitHub|https://github.com/huggingface/trl]]
* [[source::Doc|TRL SFTTrainer|https://huggingface.co/docs/trl/sft_trainer]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::Training]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Concrete tool for supervised fine-tuning (SFT) of language models provided by HuggingFace TRL library, fully compatible with Unsloth.

=== Description ===
`SFTTrainer` is the standard trainer for instruction tuning and supervised fine-tuning tasks. It extends HuggingFace's Trainer with features specific to language model fine-tuning: chat template formatting, packing, and dataset processing. When used with Unsloth-patched models, it achieves 2x faster training with full compatibility.

=== Usage ===
Import this class when performing supervised fine-tuning on instruction datasets, chat datasets, or any text-to-text task. Use after loading model with `FastLanguageModel` or `FastModel`. Standard choice for QLoRA fine-tuning workflows.

== Code Signature ==
<syntaxhighlight lang="python">
from trl import SFTTrainer

class SFTTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: Optional[SFTConfig] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        data_collator: Optional[DataCollator] = None,
        packing: bool = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        dataset_text_field: Optional[str] = None,
        **kwargs
    ):
        ...
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> TrainOutput:
        """Run training loop."""
        ...
</syntaxhighlight>

== I/O Contract ==
* **Consumes:**
    * `model`: Unsloth-patched `PreTrainedModel` with LoRA adapters
    * `train_dataset`: HuggingFace `Dataset` with text/instruction data
    * `tokenizer`: Matching tokenizer from model loading
    * `args`: `SFTConfig` with training hyperparameters
* **Produces:**
    * Training logs and metrics
    * Model checkpoints in `output_dir`
    * Final trained model state

== Example Usage ==
<syntaxhighlight lang="python">
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Configure training
args = SFTConfig(
    output_dir = "outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 10,
    max_steps = 60,
    learning_rate = 2e-4,
    logging_steps = 1,
    optim = "adamw_8bit",
    max_seq_length = 2048,
)

# Create trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = args,
)

# Train
trainer.train()
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:Unsloth_CUDA_Environment]]
* [[requires_env::Environment:Unsloth_Colab_Environment]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Gradient_Accumulation_Strategy]]
* [[uses_heuristic::Heuristic:Sequence_Packing_Optimization]]
* [[uses_heuristic::Heuristic:Learning_Rate_Tuning]]

