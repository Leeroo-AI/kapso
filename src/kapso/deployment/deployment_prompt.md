# Deploy the solution

You are deploying a finished solution as running software. The engine
checks what you deploy once you are done: the result file you write, the
files it names, the target's health check, and the call you proved, run
again.

## The solution

Its objective:

{{objective}}

Delivered: branch `{{branch}}` at commit `{{commit}}`, score {{score}}.

## The workspace

This directory is a copy of the solution, yours to adapt in place:
`{{workspace}}`. It is its own git repository. The campaign it came from
is above it and is not to be touched.

These variables are set in your environment and in every script you run:

- `KAPSO_DEPLOYMENT_NAME` = `{{deployment_name}}`: what the targets name their containers, apps and services after (`kapso-{{deployment_name}}`); use it exactly.
- `KAPSO_PORT` = `{{port}}`: the host port this deployment publishes on, when its target uses one.
- `KAPSO_NUMBER` = `{{number}}`, `KAPSO_WORKSPACE`, `KAPSO_RECORD_DIR` = `{{record_dir}}`.
- `KAPSO_INTERPRETER` = `{{interpreter}}`: the Python the engine runs the deployed software with; install the solution's requirements into it (`"$KAPSO_INTERPRETER" -m pip install -r requirements.txt`, or `uv pip install --python "$KAPSO_INTERPRETER" -r requirements.txt` when it has no pip).
- `KAPSO_TARGET_DIR`: the target's skill directory, whose `scripts/` you run (`bash "$KAPSO_TARGET_DIR/scripts/deploy.sh"`).

{{env_block}}

{{options_block}}

## The target

{{target_block}}

The target's scripts are the engine's: run them, never edit them. Export
the `KAPSO_VALUE_<KEY>` variables the skill asks for before running its
deploy script, and write the same under `values` in the result.

## The steps

1. Open the target's skill and follow it: adapt the solution to the
   target's entry point, run `scripts/deploy.sh` from this directory, fix
   what it reports and run it again until it succeeds.
2. Prove one call with real inputs: a feature row or a text the solution
   expects, not a placeholder — a row the delivered branch already scored
   when it has one (its predictions file), so the answer is held against
   that number. The proof is the delivered prediction, reproduced: a
   solution that scores its rows relative to one another (a rank blend, a
   batch normalisation) keeps the batch it was delivered with — the proof
   row is scored within it — since the same row scored against another
   reference is a different number, and a different number is no proof.
   Keep the inputs and the output.
3. Write the result file at `{{result_path}}`:

```json
{"target": "<the target's name>",
 "reasoning": "<why you chose it; required when you chose>",
 "endpoint": "<the URL, or the log path, or null>",
 "run": {"kind": "function|http|script", "...": "the kind's keys, as the skill says"},
 "values": {"<key>": "<value>", "...": "exactly what you exported as KAPSO_VALUE_<KEY>"},
 "proof": {"inputs": {"...": "the call you made"}, "output": {"...": "what it answered"}},
 "artifacts": ["<every file you wrote or changed>"],
 "summary": "<one or two sentences: what runs where, and how the proof went>"}
```

The engine then reads it (a file that is not valid JSON, or not of this
shape, comes back to you as a finding), checks that the run kind is the
target's and that the artifacts exist in the workspace, runs the target's
health check and makes the proof call itself; a check that fails comes
back to you once with the finding.

{{domain_guidance}}
