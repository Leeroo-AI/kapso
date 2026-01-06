{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Trainer Documentation|https://huggingface.co/docs/transformers/main_classes/trainer]]
* [[source::Doc|Training Tutorial|https://huggingface.co/docs/transformers/training]]
|-
! Domains
| [[domain::Training]], [[domain::Fine_Tuning]], [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
End-to-end training workflow using the Trainer class for fine-tuning or training 🤗 Transformers models from scratch with support for distributed training, mixed precision, and hyperparameter optimization.

=== Description ===
The Trainer class provides a complete training and evaluation loop optimized for 🤗 Transformers models. It handles the complexity of modern deep learning training including:

* **Distributed Training**: Support for DataParallel, DistributedDataParallel, FSDP, and DeepSpeed
* **Mixed Precision**: Automatic mixed precision training with fp16/bf16
* **Optimization**: Configurable optimizers, learning rate schedulers, and gradient accumulation
* **Checkpointing**: Automatic saving of model checkpoints with configurable strategies
* **Logging**: Integration with TensorBoard, Weights & Biases, MLflow, and other logging frameworks
* **Evaluation**: Built-in evaluation loop with customizable metrics

The Trainer abstracts away boilerplate code while remaining highly customizable through callbacks, subclassing, and configuration.

=== Usage ===
Execute this workflow when you need to:
* Fine-tune a pretrained model on a custom dataset
* Train a model from scratch with PyTorch
* Leverage distributed training across multiple GPUs/nodes
* Use advanced optimization techniques (gradient checkpointing, mixed precision)
* Integrate with experiment tracking tools

Prerequisites:
* A PreTrainedModel or compatible PyTorch model
* Training and optionally evaluation datasets
* TrainingArguments configuration

== Execution Steps ==

=== Step 1: TrainingArguments Configuration ===
[[step::Principle:huggingface_transformers_TrainingArguments_Configuration]]

Configure training hyperparameters and settings through the TrainingArguments dataclass. This centralizes all training configuration including learning rate, batch size, number of epochs, output directories, logging settings, and distributed training options.

'''Key configuration categories:'''
* Output and logging: output_dir, logging_steps, save_strategy
* Optimization: learning_rate, weight_decay, warmup_steps
* Batching: per_device_train_batch_size, gradient_accumulation_steps
* Distributed: ddp_find_unused_parameters, fsdp, deepspeed

=== Step 2: Dataset Preparation ===
[[step::Principle:huggingface_transformers_Dataset_Preparation]]

Prepare training and evaluation datasets in a format compatible with the Trainer. Datasets should be tokenized/processed and return dictionaries with input tensors. Use data collators to handle dynamic batching and padding.

'''Dataset requirements:'''
* Implement __len__ and __getitem__ (or use IterableDataset)
* Return dictionaries with model input keys
* Labels should be included for loss computation

=== Step 3: Trainer Initialization ===
[[step::Principle:huggingface_transformers_Trainer_Initialization]]

Initialize the Trainer with model, arguments, datasets, and optional components. The Trainer validates configurations, sets up the optimizer and scheduler, initializes callbacks, and prepares for distributed training if configured.

'''Initialization components:'''
* Model and tokenizer/processor
* TrainingArguments
* Train and eval datasets
* Data collator (auto-selected if not provided)
* Compute metrics function
* Callbacks for custom behavior

=== Step 4: Optimizer and Scheduler Setup ===
[[step::Principle:huggingface_transformers_Optimizer_Scheduler_Setup]]

Create the optimizer and learning rate scheduler. The Trainer supports various optimizers (AdamW, Adafactor, SGD, custom) and learning rate schedules (linear, cosine, polynomial, constant). Parameter groups can be configured to apply different learning rates or weight decay.

'''Optimizer options:'''
* AdamW (default), Adafactor, SGD
* Layer-wise learning rate decay
* Weight decay filtering (exclude bias, LayerNorm)
* Integration with PEFT for LoRA training

=== Step 5: Training Loop Execution ===
[[step::Principle:huggingface_transformers_Training_Loop]]

Execute the main training loop with forward pass, loss computation, backpropagation, and optimizer step. Handle gradient accumulation, mixed precision, gradient clipping, and distributed synchronization.

'''Training loop structure:'''
* Iterate over epochs and batches
* Forward pass with loss computation
* Backward pass with gradient scaling (if fp16)
* Gradient clipping and optimizer step
* Learning rate scheduler step
* Logging and checkpointing

=== Step 6: Evaluation Loop ===
[[step::Principle:huggingface_transformers_Evaluation_Loop]]

Run evaluation on the validation dataset at configured intervals. Collect predictions and compute metrics using the provided compute_metrics function.

'''Evaluation behavior:'''
* Run at end of epoch or every N steps
* Collect all predictions across distributed processes
* Compute custom metrics (accuracy, F1, BLEU, etc.)
* Log metrics to configured backends

=== Step 7: Checkpoint Saving ===
[[step::Principle:huggingface_transformers_Checkpoint_Saving]]

Save model checkpoints according to the save strategy. Supports saving best model, last N checkpoints, and integration with Hub for automatic uploads.

'''Checkpoint contents:'''
* Model weights (safetensors or pytorch_model.bin)
* Optimizer and scheduler states
* Trainer state (epoch, step, best metric)
* Training arguments

== Execution Diagram ==
{{#mermaid:graph TD
    A[TrainingArguments Configuration] --> B[Dataset Preparation]
    B --> C[Trainer Initialization]
    C --> D[Optimizer & Scheduler Setup]
    D --> E[Training Loop Execution]
    E --> F[Evaluation Loop]
    F --> G[Checkpoint Saving]
    F -->|Continue Training| E
}}

== Related Pages ==
* [[step::Principle:huggingface_transformers_TrainingArguments_Configuration]]
* [[step::Principle:huggingface_transformers_Dataset_Preparation]]
* [[step::Principle:huggingface_transformers_Trainer_Initialization]]
* [[step::Principle:huggingface_transformers_Optimizer_Scheduler_Setup]]
* [[step::Principle:huggingface_transformers_Training_Loop]]
* [[step::Principle:huggingface_transformers_Evaluation_Loop]]
* [[step::Principle:huggingface_transformers_Checkpoint_Saving]]
