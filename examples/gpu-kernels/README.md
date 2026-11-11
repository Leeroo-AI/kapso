# CUDA optimization: faster multi-head attention

A ninety-minute build on a CUDA machine that speeds up the vanilla
multi-head self-attention in `repo/module.py` (sequence 256, embedding 768,
8 heads, batch 32) while keeping its results within 1e-5, with the
repository's own `evaluate.py` measuring correctness and speedup. Flash
attention or a Triton kernel is the route the brief's target assumes.

## What you get

- `repo/module.py`, rewritten on a branch of its own inside `campaign/`.
- A score from `repo/evaluate.py --path module.py`: the speedup over the
  baseline, after a correctness check, printed as `Score: 2.3400`.
- `campaign/.kapso/solution.json`, the record, read back by `Solution.open("campaign")`.

## Run it

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton
kapso doctor build
kapso build --brief kapso-brief.md --repo repo --output campaign --detach
kapso watch campaign --follow
```

`python run.py` is the same build from Python, into the same `campaign/`
directory: run one or the other. A CUDA device is required.

## What to look at

- `kapso-brief.md`: the interface, the tolerance and the CUDA requirement
  are rules; the judge rejects a candidate that breaks one.
- `solution.diff()`: the kernel that won.
- `cd campaign && python evaluate.py --path module.py`: the speedup, re-measured
  by hand.
- `kapso trace campaign`: every attempt with its sessions and runs, its
  score and the judge's words, what was rejected and why, what it cost.

## Change it

- The shape: the parameters at the top of `repo/evaluate.py`
  (`max_seqlen`, `seq_len`, `n_embd`, `n_head`, `batch_size`).
- The target: a number in `Target` once you know the hardware.
- The budget: the brief's `Budget minutes=90 experiments=8`, or the command's
  `--budget-minutes` and `--budget-experiments`, which win over it.
- Research first: `kapso experience add --research "flash attention in Triton"
  --name triton-attention`, then build with `--experience triton-attention`
  so the sessions can search what the research found.
