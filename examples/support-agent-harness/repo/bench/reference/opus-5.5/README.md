# A second recorded run

Opus 5.5 as the agent through the same runner, loop, tools, policy and simulated customer as the harness, on the 20 development conversations, one trial, 2026-11-02: 8 of 20 resolved, pass_1 0.400.

- `summary.json`: the numbers, in the shape `bench/run_dev.py` writes.
- `episodes.jsonl`: one line per conversation: outcome, how it ended, counts and tokens, the calls made,
  the reference route's matched flags. No expected value is written.
- `transcripts/<conversation>.md`: the conversations as they ran, with the tool calls.
