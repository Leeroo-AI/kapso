# Trading: a task on the trading harness

This example is research software on synthetic data. It is not financial advice, and nothing in it is a recommendation to trade.

The author's example: a harness (`../harnesses/trading`), a task on it, and
the probe run an author reads to improve the harness. The task is a daily
strategy on three synthetic instruments with a trend in them; the shipped
20/100 moving-average cross scores a Sharpe of 0.14 out of sample (measured
on 2026-10-25 by the example's evaluator), the brief asks for 1.0, and a
faster cross or a volatility-targeted momentum signal gets there, so there
is room. The harness adds what a bare build
lacks: a setup session that pins the data as a prepared snapshot, pages on
timing discipline and the risk line, a look-ahead scan the implementer runs
before the backtest, an auditor the verifier asks, and hooks that deny a
web fetch or a write under the snapshot.

## What you get

- `repo/strategy.py`, the starting signal, improved on a branch of its own
  inside `campaign/`, one branch per experiment, each verified by the
  harness before it is judged.
- A score measured by `evaluation/evaluate.py`: the 2024 Sharpe after
  costs, on bars that never enter the workspace, with the drawdown and
  turnover beside it.
- `campaign/.kapso/solution.json`, the record, with the harness's identity
  and its verdict on the delivered candidate; `campaign/kapso_task.md`, the
  task statement every session read; `campaign/prepared/market-data`, the
  snapshot the setup session pinned; `campaign/.kapso/verification/<branch>/`,
  each attempt's diff, evaluation output and verdict.

## Run it

```bash
cd examples/trading
pip install numpy pandas pyarrow "mcp<2"   # the generator, the evaluator, the harness's servers (written on the mcp 1.x API)
python generate.py                      # 2021-2023 under data/, 2024 under private/, seeded
kapso harness check ../harnesses/trading
kapso build --brief kapso-brief.md --evaluation evaluation --data data --repo repo --output campaign --detach
kapso watch campaign --follow
```

`python run.py` is the same build from Python, into the same `campaign/`
directory: run one or the other. The brief's `Harness` line names the
harness, so the command needs no `--harness`; `--harness` would win over
the line.

## What to look at

- `campaign/kapso_task.md`: the brief, the budget, the references, the
  setup session's domain statement and the prepared resources, as every
  session saw them.
- `campaign/.kapso/sessions/<session>/session.json` and `stream.jsonl`:
  what each session was staged with and each run the host made for it
  (the hooks that fired and what they decided, the delegates and their
  reports, the tools, the cost), and the host's own event stream, where
  a hook's denial (`web fetches are off in this harness`) and a
  delegate's report are in full.
- `campaign/.kapso/verification/<branch>/verdict.json`: the verifier's
  verdict with its findings; a void one is why a scored attempt was not
  delivered.
- `solution.explain()`: the candidate, its score, the harness and the
  verdict; `solution.diff()`: the code against the seed.
- `kapso trace campaign`: the whole record laid out, every attempt with
  its sessions and runs, the cards it read, what was voided and on what
  evidence; `--branch generic_exp_0` for one attempt in full.

## Change it

- The harness: edit a page under `../harnesses/trading/instructions/`, bump
  `version` in its `harness.yaml`, run `kapso harness check` again, and the
  next build records the new hash; a resume of this campaign refuses it by
  name, which is the point.
- The target: edit `Target` and the objective. The data has a trend worth
  about a Sharpe of 2 to a 20-day momentum signal; a target above 3 is not
  there.
- The evaluator: `evaluation/evaluate.py` imports `signal` from the
  workspace, shifts it by one bar, charges 5 basis points per unit of
  turnover, scores 2024 as an equal-weight portfolio, writes
  `results/equity_curve.csv` and `results/trades.csv` (the harness's
  `require-results` hook asks an implementation session for them) and
  prints `sharpe: 0.1384` as its last line; `--fast` scores one instrument
  over one quarter, a smoke check. Change the metric there and in
  `Measured by` together.
- The data: `generate.py` (seed 20261025) writes three instruments with a
  trend component of 0.35 on the last twenty days' average return; the
  weight is what makes the baseline weak and the target reachable.

The out-of-sample bars are not sealed by this example: they sit outside the
directories the engine copies into the workspace, and the evaluator reads
them from there. Sealing is the engine's job; until then this example
measures the flow and the harness, not the boundary.
