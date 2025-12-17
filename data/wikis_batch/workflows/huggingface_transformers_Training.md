{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|HuggingFace Training Documentation|https://huggingface.co/docs/transformers/training]]
|-
! Domains
| [[domain::LLMs]], [[domain::Training]], [[domain::Fine_Tuning]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
End-to-end process for training and fine-tuning transformer models using the Trainer class with support for distributed training, mixed precision, and various optimization strategies.

=== Description ===
This workflow covers the complete training pipeline in the Transformers library using the `Trainer` class. It handles:

1. **Training Configuration**: Setting up TrainingArguments with hyperparameters and strategies
2. **Data Preparation**: Creating datasets and data collators for batching
3. **Trainer Initialization**: Configuring the trainer with model, data, and callbacks
4. **Training Loop Execution**: Running the training loop with all optimizations
5. **Checkpointing and Evaluation**: Saving models and computing metrics
6. **Model Export**: Saving the final trained model

The Trainer supports distributed training (DDP, FSDP, DeepSpeed), mixed precision (FP16, BF16), gradient accumulation, and 20+ optimizer options.

=== Usage ===
Execute this workflow when you need to:
* Fine-tune a pretrained model on a custom dataset
* Train a model from scratch on new data
* Perform distributed training across multiple GPUs/nodes
* Use advanced optimization strategies like DeepSpeed or FSDP
* Implement custom training logic with callbacks

== Execution Steps ==

=== Step 1: Training Arguments Configuration ===
[[step::Principle:huggingface_transformers_Training_Arguments]]

Configure the training hyperparameters and strategies via TrainingArguments. This dataclass contains 200+ fields controlling every aspect of training: learning rate, batch size, distributed settings, logging, checkpointing, and optimization.

'''Key configurations:'''
* Learning rate and scheduling (linear, cosine, polynomial decay)
* Batch size and gradient accumulation steps
* Distributed training mode (DDP, FSDP, DeepSpeed)
* Mixed precision training (fp16, bf16)
* Checkpointing strategy and frequency
* Evaluation and logging intervals

=== Step 2: Dataset Preparation ===
[[step::Principle:huggingface_transformers_Dataset_Preparation]]

Prepare training and evaluation datasets in a format compatible with the Trainer. Datasets should provide tokenized inputs with proper labels. The workflow handles both map-style and iterable datasets.

'''Data requirements:'''
* Datasets must be tokenized and padded appropriately
* Labels should be included in the dataset columns
* Unused columns are automatically removed
* Supports HuggingFace Datasets, PyTorch Dataset, and IterableDataset

=== Step 3: Data Collation Setup ===
[[step::Principle:huggingface_transformers_Data_Collation]]

Configure the data collator for dynamic batching. Data collators handle padding, label preparation for specific tasks, and any data augmentation. Different tasks require different collators (causal LM, masked LM, seq2seq).

'''Collator types:'''
* DataCollatorWithPadding: Dynamic padding for classification
* DataCollatorForLanguageModeling: MLM and CLM tasks with label shifting
* DataCollatorForSeq2Seq: Encoder-decoder models
* DataCollatorForTokenClassification: NER and tagging tasks

=== Step 4: Trainer Initialization ===
[[step::Principle:huggingface_transformers_Trainer_Initialization]]

Initialize the Trainer with the model, arguments, datasets, and optional customizations. The Trainer handles device placement, distributed wrapping, and accelerator configuration automatically.

'''Components configured:'''
* Model and tokenizer/processor
* Training and evaluation datasets
* Custom compute_metrics function
* Callbacks for custom behavior
* Optimizers (optional override)

=== Step 5: Training Loop Execution ===
[[step::Principle:huggingface_transformers_Training_Loop]]

Execute the training loop via `trainer.train()`. The loop handles forward/backward passes, gradient accumulation, optimizer steps, and all distributed synchronization. Callbacks are invoked at key lifecycle points.

'''Loop mechanics:'''
* Forward pass through model
* Loss computation (with label smoothing if configured)
* Backward pass with gradient scaling for mixed precision
* Gradient clipping and accumulation
* Optimizer and scheduler steps
* Logging and metric computation

=== Step 6: Evaluation and Checkpointing ===
[[step::Principle:huggingface_transformers_Evaluation_Checkpointing]]

Periodically evaluate on the validation set and save checkpoints. The trainer tracks metrics and can automatically select the best checkpoint based on a target metric.

'''Features:'''
* Configurable evaluation strategy (steps, epochs, no)
* Automatic best model selection
* Checkpoint saving with configurable frequency
* Hub upload support for checkpoints
* Metric logging to TensorBoard, W&B, etc.

=== Step 7: Model Export ===
[[step::Principle:huggingface_transformers_Model_Export]]

Save the final trained model and all associated artifacts. The exported model includes weights, configuration, tokenizer, and training metadata for easy reloading or deployment.

'''Outputs:'''
* Model weights (safetensors or bin format)
* Configuration files
* Tokenizer files
* Training arguments and state
* Optional Hub upload

== Execution Diagram ==
{{#mermaid:graph TD
    A[Training Arguments Configuration] --> B[Dataset Preparation]
    B --> C[Data Collation Setup]
    C --> D[Trainer Initialization]
    D --> E[Training Loop Execution]
    E --> F[Evaluation and Checkpointing]
    F --> G[Model Export]
    E -->|periodic| F
}}

== Related Pages ==
* [[step::Principle:huggingface_transformers_Training_Arguments]]
* [[step::Principle:huggingface_transformers_Dataset_Preparation]]
* [[step::Principle:huggingface_transformers_Data_Collation]]
* [[step::Principle:huggingface_transformers_Trainer_Initialization]]
* [[step::Principle:huggingface_transformers_Training_Loop]]
* [[step::Principle:huggingface_transformers_Evaluation_Checkpointing]]
* [[step::Principle:huggingface_transformers_Model_Export]]
