# PyTorch optimization: a faster module on CUDA

An hour-long build on a CUDA machine that makes the `Model` in
`repo/module.py` (a matrix multiplication, a division, a summation, a
scaling) faster than the shipped version while keeping its results within
1e-5, with the repository's own `evaluate.py` measuring correctness and
speedup.

## What you get

- `repo/module.py`, rewritten on a branch of its own inside `campaign/`:
  fused operations, `torch.compile`, in-place operations where safe, a
  Triton or CUDA kernel if the campaign goes that far.
- A score from `repo/evaluate.py --path module.py --device cuda`: the
  speedup over the baseline, after a correctness check.
- `campaign/.kapso/solution.json`, the record, read back by `Solution.open("campaign")`.

## Run it

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
kapso doctor build
kapso build --brief kapso-brief.md --repo repo --output campaign --detach
kapso watch campaign --follow
```

`python run.py` is the same build from Python, into the same `campaign/`
directory: run one or the other. A CUDA device is required:
the evaluator runs on it.

## What to look at

- `kapso-brief.md`: the rules keep the class interface, the numerical
  tolerance and the CUDA requirement; the judge rejects a faster module
  that breaks any of them.
- `solution.diff()`: what changed in `module.py`.
- `cd campaign && python evaluate.py --path module.py --device cuda`: the
  speedup, re-measured by hand.
- `kapso trace campaign`: every attempt with its sessions and runs, its
  score and the judge's words, what was rejected and why, what it cost.

## Change it

- The target: `Target` in the brief is "above 1.0x"; state a number if you
  know what the hardware allows.
- The budget: the brief's `Budget minutes=60 experiments=8`, or the command's
  `--budget-minutes` and `--budget-experiments`, which win over it.
- Protect the evaluator from the start by moving `evaluate.py` into a
  directory of its own and passing it with `--evaluation`.
