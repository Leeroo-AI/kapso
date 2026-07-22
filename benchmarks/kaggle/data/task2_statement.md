# 🤖 Robot Delivery Academy — imitation learning on an 8×8 city map

*IOAI 2026 AI Models Track — Practice Task 2 (Kaggle code competition
`ioai-2026-ai-models-track-practice-task-2`).*

A small delivery robot lives on an `8×8` grid city. Each episode it starts
somewhere on the map, picks up a package from one depot, and delivers it to
another depot. Some cells are blocked; every map is slightly different.

The training principle is **supervised imitation learning**: the data holds
solved examples of `observation → action`, and the mission is to train a
model that (1) learns from the demonstrations, (2) predicts a useful next
action from the current observation, (3) runs a complete episode and delivers
the package, (4) generalizes to unseen validation/test scenarios. The
demonstration budget is deliberately small — the stated challenge is whether
a model can learn from limited examples (the task text itself notes the task
is easy for a search algorithm; the *interesting question* is the learning
one). A single wrong action can drift the robot into states rare in the
demos, so high action accuracy does not always mean high episode success.

## Scoring

    SR = (# successful delivery episodes) / (# total episodes)

An episode succeeds if the package is delivered to the destination within
the step limit (**120 steps**).

## Data (`dataset/`)

- `train_demos.pkl` — pickle dict `{"trajectories": [...]}` with **400
  solved runs from 100 layouts** (all 400 succeed; 4–30 steps, median 13).
  Each trajectory: `layout_id`, `episode_seed`, `scenario`, `observations`
  (per step: `grid` (6,8,8), `vector` (13,), `action_mask` (6,), `state`
  compact simulator tuple), `actions` (list[int]), `success`, `num_steps`.
- `valid_scenarios.pkl` — **200 scenarios from 50 layouts**, no answers.
- `test_scenarios.pkl` — **1600 scenarios from 400 layouts**, no answers:
  these are what you submit for.
- Scenario dict: `layout_id`, `layout_seed`, `episode_seed`, `walls`
  (blocked [row,col] cells), `depots` (six [row,col] cells), `agent_pos`
  (start), `package_location` (depot index 0..5), `destination` (depot
  index 0..5). Scenarios are FULLY specified — the map, start, package and
  destination are all given.

## Actions

| id | action |
|---:|---|
| 0 | south |
| 1 | north |
| 2 | east |
| 3 | west |
| 4 | pickup |
| 5 | dropoff |

## Submission format (open-loop action sequences)

`submission.csv` with header `id,actions`; one row per test scenario, id =
`layout_id__episode_seed`, actions = a JSON list of action ids:

```csv
id,actions
test_0000__300000,"[1,1,2,4,0,5]"
```

No reference submission file is provided — construct the rows from
`test_scenarios.pkl` (keep its order). Submissions are pre-computed
sequences replayed by the grader: verify from the demos what the transition
dynamics are (the per-step `state`/`grid` chains in `train_demos.pkl` are
your ground truth for reconstructing them, and demo action sequences replay
deterministically to success in 400/400 cases).

## Binding rules (from the competition rules page)

1. **Competition data only** — no external data of any kind.
2. **No external pretrained models/weights/embeddings** (none is provided
   for this task either — anything you train starts from scratch, from the
   provided demos only). Libraries that auto-download weights are banned.
3. No external AI services/APIs inside the solution.
4. Notebook-based code competition: the submitted kernel generates
   `submission.csv` itself and **runs with Internet OFF**; it must be
   reproducible from competition data + code alone.
5. **Daily submission cap: 5 per day**; up to 2 final submissions selected.
6. The written rules restrict *resources* (data/models), not *methods*; the
   task framing above states the spirit (learning from demos). Weigh both
   when choosing the approach — and note the reproducibility review reads
   the submitted kernel.
