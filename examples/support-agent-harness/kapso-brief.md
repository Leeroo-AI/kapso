Objective    Move the Greywell Bank support agent from Fable 5.1 to our own Qwen 27B and improve the agent
             harness (harness/serve.py and the modules under harness/: Harness.prepare, Harness.review and
             SETTINGS) so that Qwen through the harness resolves our customer conversations at a rate
             similar to or better than Fable 5.1: pass_1 at or above 0.50 on the 20 development
             conversations, Fable's recorded 10 of 20 (bench/reference/fable-5.1/), with the model, its
             serving settings, the runner's loop, tools, step limits and sampling, the simulated customer
             and the conversations unchanged (bench/fixed_conditions.json). The ask: "our support
             agent runs on Fable 5.1 today, and we want to move it to our own Qwen model. improve the agent
             harness so we get similar or better quality than Fable 5.1 on our customer conversations".
Measured by  python kapso_evaluation/evaluate.py from the workspace root: bench/run_dev.py unchanged on the
             20 development conversations of bench/dev_tasks.json, one trial each, six at a time, at the
             fixed conditions. It checks the bench files, the runner's hashes and the endpoint's identity
             first, prints the paired flips against Fable 5.1 and Opus 5.5, then pass_1 last, the share of
             conversations the runner scores as resolved. One full run takes 45 to 90 minutes; --fast (five
             conversations) is a smoke check and never a measurement.
Baseline     0.40 (8 of 20): harness 0.1.0, the shipped pass-through, one full run of the 20 by this
             evaluator on 2026-11-08, 11:09 to 12:05 UTC (56.6 min, 667 s per conversation on average),
             1 error (task_102, the unreachable judge). Against Fable 5.1 it gained task_005 and lost
             task_055, task_070 and task_101 (4 flips, net -2); against Opus 5.5 net 0. Its episodes,
             summary and the evaluator's result are kept at kapso_datasets/baseline-qwen-harness-0.1.0/
             inside the build (bench/reference/qwen-harness-0.1.0/ in the repository) for paired comparison.
Target       Similar to or better than Fable 5.1: a full run at Fable's 10 of 20 or above, confirmed by a
             second full run of the same VERSION also at 10 of 20 or above. Identical builds flip 3 to 5 of
             20 conversations between runs (bench/eval_profile.md), so a run at 10 or 11 is read against the
             paired flips the evaluator prints, not the headline alone; reaching Fable's level meets the
             objective and a run above it is a clear improvement. The ceiling is 19 of 20 while task_102's
             judge is unreachable (Rules).
Rules
  - the evaluation is never modified: kapso_evaluation/ is protected, and bench/run_dev.py,
    bench/adapter.py, bench/endpoint.json, bench/dev_tasks.json, bench/protection.json and
    bench/fixed_conditions.json stay byte-identical to its protected copies
  - the fixed conditions of bench/fixed_conditions.json hold: the model and its serving settings, the
    runner's loop, tools, step limits and sampling, the simulated customer and the 20 conversations never
    change; only the harness (harness/serve.py, modules under harness/) and its SETTINGS change
  - the paths bench/protection.json lists are never read and its forbidden patterns never searched for:
    no task definition, expected state or answer enters a session, and runtime behaviour never recognises
    a task id or copies a development outcome
  - the harness contract in README.md holds: python -m harness.serve --port <port> --upstream <base url>
    starts it; POST /v1/chat/completions, GET /v1/models and GET /health answer as stated; an action comes
    back as a tool call in the reply's own format; upstream errors pass through with their status; standard
    library unless requirements.txt pins an addition; VERSION is bumped with every measured change
  - only a full run of the 20 is a measurement; --fast never is; the held-out conversations are never run
    or read inside the build
  - a result near Fable's 10 of 20 (10, 11 or 12) is read against the paired flips the evaluator prints and
    against the previous run of the same build, not the headline alone, before the objective counts as met
  - task_102 is scored by a natural-language judge (openrouter/openai/gpt-5.4-mini) that needs
    OPENROUTER_API_KEY in ~/.secrets/customer.env, which is absent: it errors on every run, is not a
    harness failure, and no attempt targets it
Budget       minutes=720 experiments=10 reserve_minutes=120
Harness      agent-engineering
