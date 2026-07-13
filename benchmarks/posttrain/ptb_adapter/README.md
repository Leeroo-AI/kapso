# PostTrainBench adapter for Kapso

Files here plug Kapso into a checkout of
[aisa-group/PostTrainBench](https://github.com/aisa-group/PostTrainBench) as a
first-class agent scaffold, following the same contract as the built-in
`agents/claude`, `agents/codex`, `agents/opencode` adapters.

## Apply to a PostTrainBench checkout

```bash
PTB=/path/to/PostTrainBench
cp -r agents/kapso        "$PTB/agents/kapso"
cp containers/kapso.def   "$PTB/containers/kapso.def"
cp build_kapso_container.sh "$PTB/"
cd "$PTB"
bash build_kapso_container.sh /path/to/kapso   # builds containers/kapso.sif
```

No changes to `src/run_task.sh` are required: kapso authenticates with
`ANTHROPIC_API_KEY`, which the harness already forwards into the container,
and the kapso entrypoint is invoked by absolute path
(`/opt/kapso/venv/bin/expert-posttrain`), so the harness's fixed `PATH` is
fine.

## Run a task on a single machine (no HTCondor)

`src/run_task.sh` is self-contained; HTCondor only schedules it. On a machine
with one H100, apptainer, and fuse-overlayfs:

```bash
export ANTHROPIC_API_KEY=...          # kapso + Claude Code
export OPENAI_API_KEY=...             # only needed for arenahardwriting/healthbench
                                      # judges and the post-run contamination judge
export POST_TRAIN_BENCH_CONTAINER_NAME=kapso
export HF_HOME=/path/to/hf_cache      # pre-populated via containers/download_hf_cache/

bash containers/build_container.sh vllm_debug    # eval container (once)
bash src/run_task.sh gsm8k kapso Qwen/Qwen3-4B-Base run001 10 claude-opus-4-6 1
#                    ^eval ^agent ^model            ^id   ^h  ^AGENT_CONFIG    ^gpus
```

Results land in `results/kapso_claude-opus-4-6_10h/gsm8k_Qwen_Qwen3-4B-Base_run001/`
(`metrics.json` is the score; `solve_out.txt` the agent trace;
`contamination_judgement.txt` / `disallowed_model_judgement.txt` the judge
verdicts — the judge step needs `OPENAI_API_KEY` and is safe to let fail on
unofficial dev runs).

## How the pieces map

| Harness contract | Kapso implementation |
|---|---|
| `agents/kapso/solve.sh` gets `$PROMPT`, `$AGENT_CONFIG`, cwd=task dir | execs `expert-posttrain` (benchmarks/posttrain/runner.py in the kapso repo) |
| 10h deadline via `timeout` + `timer.sh` | runner parses timer.sh → `orchestrator.solve(time_budget_minutes=...)`, holds back 20 min for consolidation, traps SIGTERM |
| `final_model/` required output | agent maintains best-so-far final_model continuously; runner's finally-block restores from artifacts if needed |
| "don't modify evaluate.py/templates" etc. | rules embedded verbatim in the handler context; kapso LLM roles pinned to Anthropic |
| self-evaluation via `evaluate.py --limit N` | drives each iteration's `<score>`, validated by kapso's FeedbackGenerator |
