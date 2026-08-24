# IOAI 2026 Integration

Every summer the sharpest young AI minds on the planet sit the same exam. The
[International Olympiad in Artificial Intelligence](https://ioai-official.org) is the IMO of
the AI era: at IOAI 2026 in Astana, 471 contestants from 108 countries and territories
competed on expert designed problems spanning the whole craft of machine learning, from
computer vision and language to optimization under a single GPU budget.

In 2026 the olympiad opened a second arena. [IOAI², the AI Model Track](https://ioai-official.org/ai-model-track/),
puts AI systems in the same exam hall: the same
[six contest tasks](https://github.com/IOAI-official/IOAI-2026), two fully autonomous 6 hour
sessions with three tasks each, up to 50 submission attempts per task, every solution scored
on standardized single GPU hardware. Once a session starts, no human may solve, correct, or
improve anything. Fourteen AI labs entered as Founding AI Participants; Kapso was one of them.

## Results

🎓 **Surpassed the best human performance**: the total of 536.07 topped every one of the
471 contestants from 108 countries

🥇 **Top 3 of all AI system participants**: the 14 Founding AI Participants spanned major
AI labs and several startups

🏆 **IOAI² Grand Master Trophy**

Per task:

| # | Task | Score |
|---|---|---|
| 1 | Speech understanding: reconstruct the chronological order of a shuffled spoken conversation from raw audio | 75.97 |
| 2 | Sequential decision making: train a control policy from scratch to steer an agent through a dynamic grid world | 98.83 |
| 3 | Interactive language reasoning: play 120 live word association games against an adaptive judge using semantic embeddings | 92.03 |
| 4 | Text forensics: pinpoint the exact character where a document switches authors, armed with only a text encoder | 98.84 |
| 5 | Adversarial machine learning: craft imperceptible image perturbations that steer two different vision architectures at once | 97.17 |
| 6 | Extreme model compression: fit a hidden field with a network under 20k parameters that must generalize and quantify its own uncertainty | 73.23 |
|   | **Total** | **536.07** |

## How it works

1. **Preflight**: one agent session ingests the official task brief, downloads the data, and
   writes the task statement. The campaign clock starts at brief-in.
2. **Campaign**: the Kapso platform runs its experimentation loop (ideation → implementation
   → judged feedback) in parallel lanes, each lane cycling submit-and-learn rounds through the
   official submission system: predict the score, submit, bank the result, study the gap, go
   again.
3. **Shared learning**: lanes learn from every sibling submission on the board, and ideas are
   grounded in the campaign [knowledge bank](past_learning/) distilled from past olympiad
   tasks.

## Quickstart

```bash
# from the repository root
pip install -e .

# ingest a task (URL or organizer brief) into a run root
PYTHONPATH=src:. python -m benchmarks.ioai2026.preflight \
    --task <task url or brief> --root tmp/ioai/task1

# run the campaign for one 6 hour session
PYTHONPATH=src:. python -m benchmarks.ioai2026.runner --root tmp/ioai/task1 --hours 6
```

## Layout

| path | role |
|---|---|
| `handler.py` | benchmark handler: the submit-and-learn lane contract, insured finalization |
| `runner.py` / `preflight.py` | campaign driver and staging / task ingestion |
| `kernel_slots.py` | ticket office over the submission platform's per account session limits |
| `config.yaml` | campaign mode (models, lanes, budgets, knowledge bank staging) |
| `past_learning/` | the harvest factory: runs on past olympiad tasks that feed the knowledge bank |
| `RULES.md` | organizer rules staged into every run |
