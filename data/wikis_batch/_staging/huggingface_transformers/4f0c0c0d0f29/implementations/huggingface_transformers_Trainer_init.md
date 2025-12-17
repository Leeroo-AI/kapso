{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete initialization method for the Trainer class provided by the HuggingFace Transformers library.

=== Description ===

The Trainer.__init__() method creates and configures a Trainer instance, which is the high-level training API in transformers. It accepts a model, training arguments, datasets, data collator, and optional components like metrics computation and callbacks. The initialization sets up the training environment including distributed training configuration, device placement, memory tracking, optimizer/scheduler creation, and callback registration. It validates configurations and prepares all components needed for the training loop.

=== Usage ===

Use Trainer.__init__() when you need to set up a training pipeline for transformer models. This is the standard entry point for training, evaluation, and prediction with HuggingFace models. Initialize it with your model, TrainingArguments, and datasets to create a fully-configured training manager that handles the complexity of modern deep learning training including mixed precision, distributed training, gradient accumulation, and checkpointing.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/trainer.py
* '''Lines:''' 382-800

=== Signature ===
<syntaxhighlight lang="python">
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
)
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
| model || PreTrainedModel or nn.Module || Yes (or model_init) || The model to train, evaluate, or use for predictions
|-
| args || TrainingArguments || No (default: output_dir="tmp_trainer") || The training arguments configuring the training loop
|-
| data_collator || DataCollator || No (auto-detected) || Function to collate samples into batches
|-
| train_dataset || Dataset or IterableDataset || No || The dataset to use for training
|-
| eval_dataset || Dataset or dict[str, Dataset] || No || The dataset(s) to use for evaluation
|-
| processing_class || Tokenizer or Processor || No (deprecated, use tokenizer) || The processing class for the model
|-
| model_init || Callable || No || Function that instantiates the model (for hyperparameter search)
|-
| compute_loss_func || Callable || No || Custom loss computation function
|-
| compute_metrics || Callable || No || Function to compute metrics from predictions
|-
| callbacks || list[TrainerCallback] || No || List of callbacks to customize training behavior
|-
| optimizers || tuple[Optimizer, Scheduler] || No (auto-created) || Custom optimizer and learning rate scheduler
|-
| optimizer_cls_and_kwargs || tuple || No || Custom optimizer class and its initialization arguments
|-
| preprocess_logits_for_metrics || Callable || No || Function to preprocess logits before metric computation
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| trainer || Trainer || Configured Trainer instance ready to call train(), evaluate(), or predict()
|}

== Usage Examples ==

=== Basic Usage ===
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
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare dataset
dataset = load_dataset("imdb", split="train[:1000]")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)

# Create training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_split["train"],
    eval_dataset=train_test_split["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

# Ready to train
trainer.train()
</syntaxhighlight>

=== Advanced Usage with Custom Metrics ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Define custom metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='binary'
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Initialize Trainer with custom metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
</syntaxhighlight>

=== Usage with Callbacks ===
<syntaxhighlight lang="python">
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)

# Custom callback
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}: {logs}")

# Initialize with callbacks
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        LoggingCallback()
    ]
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Trainer_Initialization]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
