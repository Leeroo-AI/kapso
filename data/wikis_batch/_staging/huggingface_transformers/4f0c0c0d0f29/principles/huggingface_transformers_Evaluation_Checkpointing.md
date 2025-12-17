{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Evaluation]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Evaluation and checkpointing are complementary processes for assessing model performance and preserving model states during training.

=== Description ===

Evaluation measures model performance on held-out data to assess generalization and guide training decisions. During training, periodic evaluation on a validation set provides feedback about whether the model is learning effectively, overfitting, or ready to stop training. Evaluation involves switching the model to inference mode (disabling dropout and batch normalization training behavior), computing predictions on validation examples, calculating metrics beyond training loss (accuracy, F1, BLEU, etc.), and comparing results to previous evaluations to track progress.

Checkpointing complements evaluation by persisting model state (parameters, optimizer state, random seeds) to disk at strategic points. This enables fault tolerance (resume after crashes), model selection (save and later retrieve the best-performing checkpoint), and reproducibility (restore exact training state). The combination of evaluation and checkpointing implements a principled approach to preventing overfitting: regularly evaluate model performance, save checkpoints when performance improves, and ultimately restore the checkpoint with best validation performance rather than the final training state.

Together, these processes transform training from a blind optimization process into a monitored, fault-tolerant procedure that optimizes for generalization rather than just training loss. They provide the feedback loop necessary for early stopping, hyperparameter tuning, and model selection.

=== Usage ===

Use evaluation periodically during training (every epoch or every N steps) to monitor generalization performance and detect overfitting. Apply checkpointing at the same intervals as evaluation, saving model state when validation metrics improve. This pattern is essential when training on limited data where overfitting is a concern, when training time is long and fault tolerance is needed, when conducting hyperparameter searches requiring comparison across runs, or when you need to preserve and deploy the best-performing model rather than the final training state.

== Theoretical Basis ==

Evaluation and checkpointing implement model selection based on generalization performance rather than training loss.

'''Evaluation Process:'''

Goal: Estimate model performance on unseen data using validation set.

Given:
* Trained model f_θ with parameters θ
* Validation dataset D_val = {(x_i, y_i)}_{i=1}^M
* Metric function μ (accuracy, F1, etc.)

Compute:
```
performance = μ({f_θ(x_i)}_{i=1}^M, {y_i}_{i=1}^M)
```

'''Evaluation Algorithm:'''
<syntaxhighlight lang="text">
function evaluate(
    model: NeuralNetwork,
    eval_dataset: Dataset,
    compute_metrics: Callable,
    batch_size: int
) -> Dict[str, float]:
    """
    Compute model performance on evaluation dataset

    Returns dictionary of metric name -> value
    """

    model.eval()  # Switch to evaluation mode
    # - Disables dropout (uses all neurons, no randomness)
    # - Batch normalization uses population statistics
    # - Prevents gradient computation (saves memory)

    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    # Iterate through evaluation data
    with torch.no_grad():  # Disable gradient tracking
        for batch in create_batches(eval_dataset, batch_size):
            # Forward pass only
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            # Accumulate predictions and labels
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

            # Accumulate loss
            total_loss += outputs.loss.item()
            num_batches += 1

    # Compute metrics
    avg_loss = total_loss / num_batches

    eval_pred = EvalPrediction(
        predictions=np.array(all_predictions),
        label_ids=np.array(all_labels)
    )

    metrics = compute_metrics(eval_pred)
    metrics["eval_loss"] = avg_loss

    model.train()  # Switch back to training mode

    return metrics


# Example metrics computation
function compute_classification_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    accuracy = (predictions == labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted'
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
</syntaxhighlight>

'''Checkpointing Algorithm:'''
<syntaxhighlight lang="text">
function checkpoint_management(
    model: NeuralNetwork,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    args: TrainingArguments
) -> None:
    """
    Save model checkpoint and manage checkpoint lifecycle
    """

    # Create checkpoint containing full training state
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "metrics": metrics,
        "rng_state": get_rng_state(),  # For reproducibility
        "args": args
    }

    checkpoint_path = f"{args.output_dir}/checkpoint-{step}"
    save_checkpoint(checkpoint, checkpoint_path)

    # Checkpoint management: keep only recent checkpoints
    if args.save_total_limit is not None:
        checkpoints = sorted(
            glob(f"{args.output_dir}/checkpoint-*"),
            key=lambda x: int(x.split('-')[-1])
        )

        # Remove oldest checkpoints beyond limit
        while len(checkpoints) > args.save_total_limit:
            oldest = checkpoints.pop(0)
            remove_checkpoint(oldest)

    # Save best model if metrics improved
    metric_value = metrics[args.metric_for_best_model]

    if is_best_metric(metric_value, args):
        save_checkpoint(
            checkpoint,
            f"{args.output_dir}/best_model"
        )

        # Update best metric tracking
        state.best_metric = metric_value
        state.best_model_checkpoint = checkpoint_path


function resume_from_checkpoint(
    checkpoint_path: str,
    model: NeuralNetwork,
    optimizer: Optimizer,
    scheduler: LRScheduler
) -> TrainingState:
    """
    Restore training state from checkpoint to resume training
    """

    checkpoint = load_checkpoint(checkpoint_path)

    # Restore all state
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    set_rng_state(checkpoint["rng_state"])

    return TrainingState(
        epoch=checkpoint["epoch"],
        global_step=checkpoint["global_step"],
        metrics=checkpoint["metrics"]
    )
</syntaxhighlight>

'''Integration: Evaluation + Checkpointing:'''
<syntaxhighlight lang="text">
function training_with_evaluation_and_checkpointing(
    model, train_data, eval_data, args
):
    best_metric = -infinity if args.greater_is_better else infinity

    for epoch in range(args.num_train_epochs):
        # Training phase
        train_one_epoch(model, train_data)

        # Evaluation phase
        if should_evaluate(epoch, args.eval_strategy):
            eval_metrics = evaluate(
                model,
                eval_data,
                compute_metrics,
                args.per_device_eval_batch_size
            )

            log_metrics(eval_metrics, step=epoch)

            # Model selection: is this the best model so far?
            current_metric = eval_metrics[args.metric_for_best_model]

            improved = (
                (args.greater_is_better and current_metric > best_metric) or
                (not args.greater_is_better and current_metric < best_metric)
            )

            if improved:
                best_metric = current_metric

                # Save best checkpoint
                checkpoint_management(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    eval_metrics,
                    args
                )

                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if (args.early_stopping_patience and
                epochs_without_improvement >= args.early_stopping_patience):
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Load best model at end
    if args.load_best_model_at_end:
        load_checkpoint(model, f"{args.output_dir}/best_model")

    return model
</syntaxhighlight>

'''Mathematical Formulation:'''

Training set performance: L_train(θ)
Validation set performance: L_val(θ)

Goal: Find θ* that minimizes L_val, not L_train

Without evaluation: θ* = argmin_θ L_train(θ)
* Risk: overfitting, L_train(θ) << L_val(θ)

With evaluation:
```
θ_t = train_step(θ_{t-1}, train_data)
if t % eval_interval == 0:
    metric_t = evaluate(θ_t, val_data)
    if metric_t > best_metric:
        θ_best = θ_t
        best_metric = metric_t

return θ_best  # Not θ_final
```

This implements a form of implicit regularization through model selection.

'''Evaluation Metrics Beyond Loss:'''

Classification:
* Accuracy = (TP + TN) / (TP + TN + FP + FN)
* Precision = TP / (TP + FP)
* Recall = TP / (TP + FN)
* F1 = 2 × (Precision × Recall) / (Precision + Recall)

Sequence Generation:
* BLEU: n-gram overlap with references
* ROUGE: recall of n-grams
* Perplexity: exp(cross-entropy loss)

Ranking:
* Mean Reciprocal Rank (MRR)
* Normalized Discounted Cumulative Gain (NDCG)

'''Checkpoint Storage:'''

Full checkpoint contains:
* Model parameters θ (~100MB to 10GB+ depending on model size)
* Optimizer state (momentum buffers, ~2x model size for Adam)
* Scheduler state
* Training step counter
* Random seeds (for reproducibility)
* Training arguments (for documentation)

Storage strategy: Keep last N checkpoints + best checkpoint

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Trainer_evaluate]]
