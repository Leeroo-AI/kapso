{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Trainer Documentation|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete training orchestrator class that unifies model training, evaluation, and prediction provided by HuggingFace Transformers.

=== Description ===
The Trainer class implements the trainer initialization principle by providing a comprehensive training orchestrator that accepts a model, training configuration, datasets, and various optional components, then coordinates their interaction throughout the training lifecycle. During initialization, it validates all component compatibility, sets up the execution environment, configures device placement and distributed training, and prepares all state tracking and callback mechanisms.

This implementation handles complex scenarios including distributed training across multiple GPUs, mixed precision training, gradient accumulation, model parallelism, and integration with various logging and experiment tracking frameworks. It automatically manages memory, sets random seeds for reproducibility, and provides hooks for customization through callbacks.

=== Usage ===
Import and instantiate Trainer after creating your model, training arguments, and datasets. Minimum requirement is a model and output directory (via args). Commonly pass train_dataset, eval_dataset, data_collator, and compute_metrics. Use after preparing all components but before calling train(). The Trainer manages all subsequent operations including training loops, evaluation, checkpointing, and prediction.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/trainer.py:L285-770

=== Signature ===
<syntaxhighlight lang="python">
class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.

        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.

        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `processing_class` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise if the processing_class is a feature extractor or tokenizer.

        train_dataset (Union[`torch.utils.data.Dataset`, `torch.utils.data.IterableDataset`, `datasets.Dataset`], *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed.

        eval_dataset (Union[`torch.utils.data.Dataset`, dict[str, `torch.utils.data.Dataset`], `datasets.Dataset`]), *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.

        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.

        model_init (`Callable[[], PreTrainedModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
            from a new instance of the model as given by this function.

        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values.

        callbacks (List of [`TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks.

        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.

        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired.
    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = None,
        args: TrainingArguments | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Union[Dataset, IterableDataset, "datasets.Dataset"] | None = None,
        eval_dataset: Union[Dataset, dict[str, Dataset], "datasets.Dataset"] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        model_init: Callable[..., PreTrainedModel] | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import Trainer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel or nn.Module || Yes* || Model to train (*or provide model_init instead)
|-
| args || TrainingArguments || No || Training configuration (defaults to basic config with output_dir="tmp_trainer")
|-
| data_collator || DataCollator || No || Function to batch samples (defaults to DataCollatorWithPadding if tokenizer provided)
|-
| train_dataset || Dataset || No || Training dataset (required for training)
|-
| eval_dataset || Dataset or dict || No || Evaluation dataset(s)
|-
| processing_class || Tokenizer or Processor || No || Tokenizer/processor for data preparation, saved with model
|-
| model_init || Callable || No || Function returning model instance (alternative to model, enables hyperparameter search)
|-
| compute_metrics || Callable || No || Function to compute evaluation metrics from EvalPrediction
|-
| callbacks || list[TrainerCallback] || No || Custom callbacks for training lifecycle events
|-
| optimizers || tuple || No || (optimizer, scheduler) tuple (defaults to AdamW with linear warmup)
|-
| optimizer_cls_and_kwargs || tuple || No || (optimizer_class, kwargs) for custom optimizer initialization
|-
| preprocess_logits_for_metrics || Callable || No || Function to process logits before metric computation
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| trainer || Trainer || Initialized trainer object ready to execute train(), evaluate(), or predict()
|}

== Usage Examples ==

=== Basic Trainer Initialization ===
<syntaxhighlight lang="python">
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare dataset
dataset = load_dataset("glue", "mrpc")
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Configure training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy="epoch"
)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator
)

# Now ready to train
trainer.train()
</syntaxhighlight>

=== Trainer with Custom Metrics ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_metric
import numpy as np

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Define custom metric computation
accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    """Compute multiple metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # Use F1 for model selection
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,  # Custom metrics
    processing_class=tokenizer  # Will be saved with model
)
</syntaxhighlight>

=== Trainer with Custom Optimizer ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./custom-optimizer-results",
    num_train_epochs=5,
    per_device_train_batch_size=8
)

# Create custom optimizer with specific parameters
optimizer = AdamW(
    model.parameters(),
    lr=3e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Create custom learning rate scheduler
num_training_steps = len(train_dataset) // 8 * 5  # total steps
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps
)

# Pass custom optimizer and scheduler to Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizers=(optimizer, lr_scheduler)  # Custom optimizer and scheduler
)
</syntaxhighlight>

=== Trainer with Callbacks ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import EarlyStoppingCallback

# Custom callback for monitoring
class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} completed")
        print(f"Current learning rate: {state.log_history[-1].get('learning_rate', 'N/A')}")

training_args = TrainingArguments(
    output_dir="./results-with-callbacks",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),  # Stop if no improvement
        CustomCallback()  # Custom monitoring
    ]
)
</syntaxhighlight>

=== Trainer with Model Init for Hyperparameter Search ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

def model_init():
    """Function that returns a new model instance."""
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    )

training_args = TrainingArguments(
    output_dir="./hp-search-results",
    eval_strategy="epoch"
)

# Use model_init instead of model for hyperparameter search
trainer = Trainer(
    model_init=model_init,  # Function instead of model instance
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Can now run hyperparameter search
best_trial = trainer.hyperparameter_search(
    direction="maximize",
    n_trials=10,
    backend="optuna"
)
</syntaxhighlight>

=== Minimal Trainer Setup ===
<syntaxhighlight lang="python">
from transformers import Trainer, AutoModelForCausalLM
from datasets import load_dataset

# Minimal setup - just model and dataset
model = AutoModelForCausalLM.from_pretrained("gpt2")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

# Trainer with defaults (args will use tmp_trainer as output_dir)
trainer = Trainer(
    model=model,
    train_dataset=dataset
)

# Ready to train with all defaults
trainer.train()
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Trainer_Initialization]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Training_Environment]]
