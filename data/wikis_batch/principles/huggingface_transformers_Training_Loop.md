{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

The training loop is the iterative process of repeatedly computing model predictions, calculating loss, computing gradients, and updating model parameters to minimize the loss function.

=== Description ===

The training loop is the fundamental algorithm for optimizing neural network parameters through gradient-based learning. It implements the core cycle of supervised learning: present training examples to the model, measure prediction errors against true labels, compute how to adjust parameters to reduce errors, and apply those adjustments. This cycle repeats for multiple passes through the training data (epochs) until the model converges to good performance.

Modern training loops incorporate numerous optimizations and features beyond basic gradient descent. They implement mini-batch training for computational efficiency, gradient accumulation to simulate larger batch sizes than fit in memory, gradient clipping to prevent training instability, learning rate scheduling to improve convergence, mixed-precision training for speed, distributed training coordination across multiple devices, periodic evaluation on validation data, checkpointing for fault tolerance, and logging for monitoring. The loop also manages state transitions between training and evaluation modes, handles data shuffling and sampling, and coordinates callbacks for custom behavior at various points in training.

The training loop bridges the gap between static model architecture and dynamic learning. It transforms a randomly-initialized model into one that has learned patterns from data by systematically applying the backpropagation algorithm guided by an optimization strategy. The quality and efficiency of the training loop implementation significantly impacts final model performance, training time, and resource utilization.

=== Usage ===

Use a training loop whenever you need to optimize model parameters from data through supervised learning. Execute a training loop after you've prepared your data, initialized your model and optimizer, and configured training hyperparameters. The training loop runs for a fixed number of epochs or until convergence criteria are met (early stopping). Apply this pattern for fine-tuning pre-trained models on new tasks, training models from scratch, conducting hyperparameter searches, or any scenario where you need to fit a parameterized model to training data through iterative optimization.

== Theoretical Basis ==

The training loop implements stochastic gradient descent (SGD) or its variants (Adam, AdamW, etc.) through iterative parameter updates.

'''Objective:'''

Minimize empirical risk over training data:
```
θ* = argmin_θ L(θ) = argmin_θ (1/N) Σ_{i=1}^N ℓ(f_θ(x_i), y_i)
```

where:
* θ = model parameters
* f_θ = model function
* ℓ = loss function
* (x_i, y_i) = training examples
* N = dataset size

'''Gradient Descent Update:'''
```
θ_{t+1} = θ_t - α_t ∇_θ L(θ_t)
```

where:
* α_t = learning rate at step t
* ∇_θ L = gradient of loss with respect to parameters

'''Mini-Batch SGD:'''

Computing gradient over entire dataset is expensive. Instead, use mini-batches:
```
∇_θ L(θ) ≈ (1/B) Σ_{i∈B} ∇_θ ℓ(f_θ(x_i), y_i)
```

where B is a random batch of size B << N.

'''Training Loop Algorithm:'''
<syntaxhighlight lang="text">
function training_loop(
    model: NeuralNetwork,
    train_data: Dataset,
    eval_data: Dataset,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    args: TrainingArguments
) -> TrainOutput:
    """
    Execute complete training process with optimization and evaluation
    """

    # Initialize
    global_step = 0
    best_metric = -infinity if args.greater_is_better else infinity
    epochs_without_improvement = 0

    # Training loop over epochs
    for epoch in range(args.num_train_epochs):
        model.train()  # Set to training mode (enables dropout, etc.)
        epoch_loss = 0

        # Inner loop over batches
        for step, batch in enumerate(train_data):
            # 1. Forward pass: compute predictions and loss
            with autocast(enabled=args.fp16):  # Mixed precision
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                loss = compute_loss(outputs, batch["labels"])

                # Scale loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps

            # 2. Backward pass: compute gradients
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()  # ∇_θ L

            # 3. Gradient accumulation: only update every K steps
            if (step + 1) % args.gradient_accumulation_steps == 0:

                # 4. Gradient clipping: prevent exploding gradients
                if args.max_grad_norm is not None:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm
                    )

                # 5. Optimizer step: θ_{t+1} = θ_t - α_t ∇_θ L
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # 6. Update learning rate
                scheduler.step()  # α_{t+1} = schedule(t)

                # 7. Reset gradients for next accumulation
                optimizer.zero_grad()

                global_step += 1

                # 8. Logging
                if global_step % args.logging_steps == 0:
                    log_metrics({
                        "loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": global_step
                    })

                # 9. Evaluation
                if should_evaluate(global_step, args):
                    eval_metrics = evaluate(model, eval_data)
                    log_metrics(eval_metrics)

                    # Check for improvement
                    metric_value = eval_metrics[args.metric_for_best_model]
                    if is_improvement(metric_value, best_metric, args):
                        best_metric = metric_value
                        save_checkpoint(model, optimizer, "best_model")
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    # Early stopping
                    if (args.early_stopping_patience is not None and
                        epochs_without_improvement >= args.early_stopping_patience):
                        print(f"Early stopping after {epoch} epochs")
                        break

                # 10. Checkpointing
                if should_save(global_step, args):
                    save_checkpoint(
                        model,
                        optimizer,
                        f"checkpoint-{global_step}"
                    )

                # 11. Check if max steps reached
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            epoch_loss += loss.item()

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_data)
        print(f"Epoch {epoch}: avg_loss={avg_epoch_loss}")

    # Load best model if requested
    if args.load_best_model_at_end:
        load_checkpoint(model, "best_model")

    return TrainOutput(
        global_step=global_step,
        training_loss=avg_epoch_loss,
        metrics={"best_metric": best_metric}
    )
</syntaxhighlight>

'''Key Mathematical Components:'''

**Gradient Computation (Backpropagation):**
```
∂L/∂θ = ∂L/∂y_pred × ∂y_pred/∂θ
```
Computed automatically via chain rule through computation graph.

**Learning Rate Schedule:**

Common schedules:
* Linear decay: α_t = α_0 × (1 - t/T)
* Cosine annealing: α_t = α_0 × (1 + cos(πt/T)) / 2
* Step decay: α_t = α_0 × γ^(t/S)
* Warmup + decay: ramp up then decay

**Gradient Clipping:**
```
if ||g|| > threshold:
    g = g × (threshold / ||g||)
```

**Momentum (Adam optimizer):**
```
m_t = β_1 × m_{t-1} + (1 - β_1) × g_t        # First moment
v_t = β_2 × v_{t-1} + (1 - β_2) × g_t^2      # Second moment
m̂_t = m_t / (1 - β_1^t)                       # Bias correction
v̂_t = v_t / (1 - β_2^t)
θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)
```

**Mixed Precision Training:**

Use FP16 for forward/backward, FP32 for optimizer:
* Speeds up computation (especially on modern GPUs)
* Reduces memory usage (~2x reduction)
* Requires loss scaling to prevent underflow

'''Convergence Criteria:'''

Training stops when:
1. Maximum epochs reached
2. Maximum steps reached
3. Early stopping triggered (no improvement for K evaluations)
4. Validation loss diverges (training failure)

'''Checkpointing Strategy:'''

Save model state at intervals to enable:
* Resume training after interruption
* Recover best model (avoid overfitting)
* Experiment comparison and model selection

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Trainer_train]]
