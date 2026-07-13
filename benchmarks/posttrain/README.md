# PostTrainBench × Kapso

Run Kapso as an agent scaffold on [PostTrainBench](https://posttrainbench.com/)
([paper](https://arxiv.org/abs/2603.08640),
[repo](https://github.com/aisa-group/PostTrainBench)): autonomously post-train
a small base LLM for a target benchmark, on one H100, in 10 hours.

Contents of this directory:

| Path | What it is |
|---|---|
| `runner.py` (`expert-posttrain`) | Kapso runner for one task: wires the 10h wall-clock budget into `OrchestratorAgent.solve`, parses the harness prompt/timer, guarantees `final_model/` on exit |
| `handler.py` | `PostTrainBenchHandler`: official rules + operational discipline fed to the coding agent each iteration |
| `config.yaml` | `POSTTRAIN` mode: generic (sequential, parent=best) search, Claude Code via `ANTHROPIC_API_KEY`, all LLM roles pinned to Anthropic, multi-hour timeouts |
| `ptb_adapter/` | Files to drop into a PostTrainBench fork: `agents/kapso/solve.sh`, `containers/kapso.def`, build script |
| `gcp/` | GCP provisioning: bootstrap, one-time asset build (containers + HF-cache disk snapshot), per-run flex-start/spot H100 VM, results fetch |

---

## 1. The benchmark, precisely

One run = (base model, target benchmark). The agent is dropped into a
container at `~/task` containing `evaluate.py` (read-only), `templates/`
(read-only eval chat templates), `timer.sh`, and optional `task_context/`
files; it gets the official prompt via stdin/env, unrestricted internet, a
shared read-only HF cache (writes go to an overlay), and one H100 for
`{num_hours}` (10) hours. It must leave its best model in `task/final_model/`,
loadable in the unmodified starting environment.

- **Base models** (always the *base/pt* variants):
  `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base`, `HuggingFaceTB/SmolLM3-3B-Base`,
  `google/gemma-3-4b-pt` (gated — needs an HF license acceptance).
- **Benchmarks** (eval task ids): `aime2025`, `arenahardwriting`, `bfcl`,
  `gpqamain`, `gsm8k`, `healthbench`, `humaneval`.
- **Scoring**: after the run, the harness evaluates `final_model` with the full
  test set (`--limit -1`), retrying with reduced max-tokens on failure.
  Everything runs through [Inspect AI](https://inspect.aisi.org.uk/) + vLLM
  (temp 0.6, top-p 0.95, `gpu_memory_utilization=0.3`) with a **fixed chat
  template per model family** (`templates/*.jinja`). ArenaHard-Writing and
  HealthBench are GPT-judge scored (needs `OPENAI_API_KEY` at eval time).
- **Aggregate**: weighted mean over all 28 model×benchmark combos, weights
  ∝ 1/(instruct − base) headroom (`scripts/factors.json`):
  AIME .227, GPQA .225, HealthBench .184, HumanEval .106, GSM8K .094,
  ArenaHard .090, BFCL .075.
- **Anti-cheating**: after the agent exits, a Codex-based judge audits the full
  trace + workspace for test-data contamination and model substitution
  (`src/disallowed_usage_judge/`). Flagged runs score as the *base model*.
  Forbidden: training on benchmark test data, modifying `evaluate.py` or
  `templates/`, fine-tuning anything but the assigned base model, using the
  OpenAI API for anything but `evaluate.py`.
- **Reference points** (leaderboard, mid-2026): base models ≈ 7.5 avg,
  best agents ≈ 23–34 (Opus 4.6/Claude Code 23.2; GLM 5.2 and Opus 4.8 ≈ 34),
  official instruct models 51.1, human post-trained ≈ 62.

### Harness mechanics (what actually runs)

`src/run_task.sh <eval> <agent> <model> <run_id> <hours> <agent_config> [gpus]`
is fully self-contained — HTCondor is only used to schedule it, so a single
GCP VM can run it directly:

1. Builds a job dir; copies in `evaluate.py`, `templates/`, `task_context/`,
   generates `timer.sh`, renders the prompt.
2. Mounts the host HF cache through **fuse-overlayfs** (agent writes land in a
   throwaway upper layer) and starts the **apptainer** container (`-c`,
   `--home job_dir:/home/ben`, `--pwd /home/ben/task`, network shared).
3. Runs `agents/<agent>/solve.sh` with `$PROMPT`/`$AGENT_CONFIG`, under
   `timeout` of hours+5min (TERM, then KILL).
4. Runs the contamination judge, then full evaluation (up to 3 token-budget
   phases × retries) producing `metrics.json`.

Agent contract = one `solve.sh` that runs your CLI non-interactively until
done. Kapso's is `ptb_adapter/agents/kapso/solve.sh`.

---

## 2. Why Kapso fits, and the integration design

PostTrainBench is exactly Kapso's `.evolve()` shape: research → design an
experiment → implement → train → self-evaluate (`evaluate.py --limit N`) →
select parent → iterate, under a budget. The official "reprompt" scaffold
variants (loop "you still have Xh, keep improving") are a crude version of
what the orchestrator does natively — and reprompted variants score higher on
their leaderboard, which suggests structured iteration pays.

Design decisions (and the kapso gaps they close):

- **Generic sequential strategy, `parent_policy: best`** — one experiment at a
  time (one GPU), each branching from the best-scoring node.
  `benchmark_tree_search`'s parallel expansion doesn't fit a single GPU;
  revisit for multi-GPU ablations.
- **Wall-clock budget**: `Kapso.evolve()`/CLI don't expose
  `time_budget_minutes`, so `runner.py` calls `OrchestratorAgent.solve()`
  directly (the MLE/ALE runner pattern) with the deadline parsed from
  `timer.sh` minus a 20-min consolidation reserve, and traps SIGTERM so the
  finally-block still runs when the harness kills us.
- **No-OpenAI compliance**: default kapso config routes utility/reasoning/web
  roles to OpenAI models — forbidden here. `config.yaml`'s POSTTRAIN mode pins
  every role to Anthropic and disables the `research` and `leeroopedia` gates
  (Claude Code's built-in WebSearch/WebFetch covers research and is allowed).
- **Weights out of git**: kapso's workspace is a git repo with per-experiment
  branches; multi-GB safetensors in it would explode the 400 GB job disk. The
  handler mandates all weights under `task/artifacts/` + a `.gitignore`.
- **Deadline-proof deliverable**: the agent must keep `task/final_model/` as
  best-so-far at all times (atomic swap + `best_score.log`); the runner's
  finally-block restores it from `artifacts/` if it's ever missing. A hard
  kill at any moment still leaves a valid submission. (Paper failure modes
  this kills: early termination, submitting a broken late experiment.)
- **Isolation of environments**: kapso + its deps (aider-chat pins, litellm,
  weaviate-client…) live in `/opt/kapso/venv` inside the container; the pinned
  training/eval stack (`transformers 4.57.3`, `trl 0.27.2`, `vllm 0.11.0`,
  `flash-attn 2.8.3`) stays untouched.
- **Timeouts**: coding-agent/session timeouts raised to 5 h (default 600 s
  would kill mid-training); `BASH_MAX_TIMEOUT_MS=36000000` for Claude Code's
  bash tool.
- **Resume**: `--resume` maps to kapso's checkpoint restore
  (`.kapso/run_state.json`) — a preempted Spot run can continue with whatever
  time `timer.sh` still grants. No other PostTrainBench scaffold can do this.

What the handler teaches the agent beyond the official prompt (all
scaffold-level, no test data): match the eval jinja template exactly when
rendering SFT data; establish the base-model score first; per-benchmark
`--limit` for cheap iteration evals; GPU exclusivity between train and eval;
prior-run insights from the paper (SFT/TRL dominates; BFCL and GSM8K have the
most headroom; GPQA fails on answer formatting; don't sink everything into
AIME).

---

## 3. GPU on GCP: options and cost

Single H100-80GB = `a3-highgpu-1g` (26 vCPU, 234 GB RAM, bundled local SSD,
gVNIC required). **It cannot be bought on-demand** — only:

| Channel | Price (us-central1-ish) | Preemptible? | Use for |
|---|---|---|---|
| **DWS Flex-start** (`--provisioning-model=FLEX_START`) | ~53% off on-demand ≈ **$5.2/h** | **No** — queued start, then runs to completion (≤7 days) | scored runs (default here) |
| **Spot** | ~$4.3–6/h (region-dependent, floats) | Yes, anytime | dev iteration (kapso can `--resume`) |
| a3-highgpu-**8g** on-demand | ~$88/h (8 GPUs) | No | not needed (benchmark is 1 GPU) |

Both Spot and Flex-start consume the **Preemptible NVIDIA H100 GPUs** quota —
request ≥1 in your region (console → Quotas), usually auto-approved.

**Cost per scored run**: ~12 h VM (10 h agent + eval + judge + overhead)
× $5.2 ≈ **$65** + ~$2 disks + Anthropic API spend (typically $30–150/run for
10 h of Claude Code — often rivals the compute cost; a Claude Max subscription
via `CLAUDE_CODE_OAUTH_TOKEN` is the cheap alternative). Full 28-combo sweep
≈ **$1.9k** compute at flex-start prices, plus API/judge costs. GPT-judged
benchmarks additionally burn OpenAI credits per eval (~$1–5 per full eval).

Architecture (implemented in `gcp/`):

```
00_bootstrap.sh      APIs, bucket, service account, secrets (anthropic/openai/hf)
01_build_assets.sh   cheap spot CPU VM builds kapso.sif + vllm_debug.sif -> GCS,
                     fills a pd disk with the HF cache -> snapshot  (one-time)
10_launch_run.sh     per run: disk-from-snapshot + a3-highgpu-1g flex-start VM
                     (ubuntu 22.04 + driver + apptainer at boot, local SSD as /tmp,
                      results rsync'd to GCS every 10 min, self-deleting)
20_fetch_results.sh  pull + summarize metrics.json / judge verdicts
```

Why the snapshot: the official cache (`containers/download_hf_cache/`,
~300 datasets + 14 models) is hundreds of GB; per-run VMs clone the snapshot
in minutes instead of re-downloading. `CACHE_SCOPE=core` (default) caches just
the 4 base models (~50 GB) — fine for dev, since the agent has internet and
can pull datasets on demand; use `full` for official-parity runs.

---

## 4. Runbook

```bash
# 0. once: fork PostTrainBench, apply ptb_adapter/ (see ptb_adapter/README.md),
#    push; set PTB_REPO_URL/KAPSO_REPO_URL (+ optionally REGION/ZONE) in gcp/env.sh
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... HF_TOKEN=...
bash gcp/00_bootstrap.sh            # + request preemptible-H100 quota in console
bash gcp/01_build_assets.sh         # ~1-2h once

# 1. smoke run (cheap, 1 agent-hour, spot): does the plumbing work end to end?
bash gcp/10_launch_run.sh gsm8k Qwen/Qwen3-1.7B-Base --hours 1 --spot
bash gcp/20_fetch_results.sh        # list run ids
bash gcp/20_fetch_results.sh <run_id>

# 2. scored run (flex-start, full 10h)
bash gcp/10_launch_run.sh gsm8k Qwen/Qwen3-4B-Base --hours 10 --agent-config claude-opus-4-6

# 3. sweep = loop 10_launch_run.sh over the 7x4 matrix (mind quota: 1 GPU per
#    concurrent run; runs are independent, so quota N => N in parallel)
```

Checklist for reading a run: `metrics.json` (score), `time_taken.txt` (should
be ~hours, not minutes — early exit means something broke),
`solve_out.txt` (agent trace), `contamination_judgement.txt` /
`disallowed_model_judgement.txt` (must be clean), `final_model/` present.

Local dev without GCP: any H100 box with apptainer + fuse-overlayfs runs
`bash src/run_task.sh ...` directly — see `ptb_adapter/README.md`.

---

## 5. Getting an official leaderboard row

There is **no public submission pipeline**: the leaderboard is curated by the
maintainers (ELLIS Tübingen / MPI-IS; contacts in the PostTrainBench README —
Ben Rank, Hardik Bhatnagar, Maksym Andriushchenko). Their repo says
contributions are welcome via PR/issue/email, and they have been adding
third-party scaffolds (glm5, qwen3max, opencode variants). Their
[Harbor support PR #8](https://github.com/aisa-group/PostTrainBench/pull/8)
(Modal/Daytona cloud runs) is still open, so HTCondor/our-VM remains the
practical path today.

Suggested sequence:
1. Validate locally: full 10 h runs on 2–3 combos on our GCP setup, clean
   judge verdicts, plausible scores.
2. Open a PR adding `agents/kapso/` + `containers/kapso.def` (exactly the
   files in `ptb_adapter/`) and email the maintainers proposing a
   "Kapso (Claude Opus …)" scaffold row — offer our Anthropic key or ask them
   to run on theirs, mirroring how other scaffolds were evaluated (3 seeds ×
   28 combos on their cluster).
3. Keep our GCS results as the reproducibility bundle for their audit.

## 6. Known gaps / next steps

- `benchmarks/posttrain/` has been syntax-checked but needs a live smoke test
  (no GPU/API keys in the authoring environment): first on a laptop with a fake
  `evaluate.py` echoing a score, then the 1 h GCP spot run above.
- Verify the FeedbackGenerator accepts the `<score>`/`result.json` convention
  end-to-end in POSTTRAIN mode, and that checkpoint resume works mid-campaign.
- `run_startup.sh` assumes `nvidia-driver-570-server` (falls back to 550) and
  the apptainer PPA on Ubuntu 22.04 — pin down on first boot; if a3 creation
  complains about local SSD, add `--local-ssd=interface=NVME` twice.
- Tree-search mode (`benchmark_tree_search`) with eval-limit-based pruning is
  the natural v2 once the sequential loop is validated; also worth testing:
  `--agent-config` sweeps (Opus vs Sonnet), and Claude Max OAuth (token file →
  `CLAUDE_CODE_OAUTH_TOKEN`) to decouple API spend from compute.
- Multi-GPU (their 8×H100/100 h ablation exists in `commit.sh`) would need
  `num_gpus` plumbed into the handler context — trivial, but not the official
  leaderboard setting.
