{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|HF Evaluate|https://huggingface.co/docs/evaluate]]
|-
! Domains
| [[domain::Model_Merging]], [[domain::Evaluation]], [[domain::Multi_Task]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Pattern for evaluating merged adapter quality by testing on tasks from each source adapter.

=== Description ===

This is a Pattern Doc - there is no single PEFT API for evaluation. Users implement evaluation loops using standard evaluation libraries (evaluate, datasets) combined with PEFT's adapter switching.

=== Usage ===

Implement a custom evaluation loop that:
1. Activates the merged adapter
2. Runs inference on test sets from each source task
3. Computes task-specific metrics
4. Compares to individual adapter baselines

== Code Reference ==

=== Source Location ===
* '''Type:''' Pattern Doc (user-implemented)
* '''No specific PEFT API''' - uses standard evaluation patterns

=== Pattern ===
<syntaxhighlight lang="python">
from evaluate import load
from peft import PeftModel

def evaluate_merged_adapter(model, merged_name, task_datasets):
    """
    Evaluate merged adapter on multiple task datasets.

    Args:
        model: PeftModel with merged adapter loaded
        merged_name: Name of the merged adapter
        task_datasets: Dict mapping task names to (dataset, metric_name)

    Returns:
        Dict of task -> metric value
    """
    model.set_adapter(merged_name)
    model.eval()

    results = {}
    for task_name, (dataset, metric_name) in task_datasets.items():
        metric = load(metric_name)

        for batch in dataset:
            with torch.no_grad():
                outputs = model.generate(batch["input_ids"], max_new_tokens=50)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        results[task_name] = metric.compute()

    return results
</syntaxhighlight>

== Usage Examples ==

=== Comparing Merged vs Individual ===
<syntaxhighlight lang="python">
# Load model with individual adapters
model.load_adapter("task1_adapter", adapter_name="task1")
model.load_adapter("task2_adapter", adapter_name="task2")

# Create merged adapter
model.add_weighted_adapter(
    adapters=["task1", "task2"],
    weights=[0.5, 0.5],
    adapter_name="merged"
)

# Evaluate individual adapters
model.set_adapter("task1")
task1_individual = evaluate_on_task1(model)

model.set_adapter("task2")
task2_individual = evaluate_on_task2(model)

# Evaluate merged adapter
model.set_adapter("merged")
task1_merged = evaluate_on_task1(model)
task2_merged = evaluate_on_task2(model)

# Compare results
print(f"Task1: individual={task1_individual:.3f}, merged={task1_merged:.3f}")
print(f"Task2: individual={task2_individual:.3f}, merged={task2_merged:.3f}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Merge_Evaluation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
