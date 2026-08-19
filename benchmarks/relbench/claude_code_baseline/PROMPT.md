You are solving one RelBench predictive task end to end: **{dataset}/{task}**.

Data and contract are in `kapso_datasets/` — read its `README.md` and
`CONTRACT.md` first. Load everything through its `load_task()` (relational
database plus train/val/test task tables) and write predictions only via
`save_predictions()`. Deliverable: `val_predictions.npy` and
`test_predictions.npy` in the contract's shape and dtype.

Test labels are masked and you must not look at, probe for, or use the test
data in any way during the whole process — only its entity/time seed rows
may be read to produce predictions. Val labels are available for your own
validation; validate honestly (time-aware where relevant) and use only the
provided data.

Budget: {hours}h wall-clock on {hardware}. The session is killed at the
deadline and whatever is on disk then is scored. Goal: maximize the task's
official test metric.
