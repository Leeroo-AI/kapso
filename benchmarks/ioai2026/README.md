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

Kapso solved all six contest tasks, and its total passed the score of the best human
contestant:

| # | Task | Score |
|---|---|---|
| 1 | Find the Order | 75.97 |
| 2 | Robot Chasing | 98.83 |
| 3 | Potato | 92.03 |
| 4 | Double Agent Dilemma | 98.84 |
| 5 | Ghost of the Machine | 97.17 |
| 6 | IOAI Field | 73.23 |
|   | **Mean** | **89.3** |

🥇 **Top 3** among all participating AI systems

🏆 **IOAI² Grand Master Trophy**

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
| `RULES.md` / `CAMPAIGN_NOTES.md` | organizer rules staged into every run / campaign engineering notes |
