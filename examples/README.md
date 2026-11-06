# Examples

Every example is a task: a brief (`kapso-brief.md`), a starting repository
(`repo/`), the data or how to get it, and `run.py`, the same build the one
command runs. `churn-first-build`, `pipeline-dbt` and `trading` carry a
seeded data generator and a protected evaluator (`evaluation/`);
`fraud-no-evaluator` carries the generator and lets the campaign build the
evaluator from the brief's protocol. The four domain examples run today
with their evaluator inside the repository, where the judge protects it by
rule; a harness for each is a follow-up. `harnesses/` holds the harness the
trading example builds on. Install Kapso and log the coding-agent CLIs in
first (`kapso doctor build` must pass), then run from the example's
directory.

Every example runs the experimentation loop; `full-loop` adds the learning
loop and the deployment, and `trading` runs on a harness, the domain
adaptation.

| Example | Minutes | What it shows | The one command | Result |
|---|---|---|---|---|
| [churn-first-build](churn-first-build/README.md) | 45 | your first build: an evaluator present, a baseline, a target, the record | `kapso build --brief kapso-brief.md --evaluation evaluation --data data --repo repo --output campaign --detach` | recorded in its README |
| [fraud-no-evaluator](fraud-no-evaluator/README.md) | 45 | no evaluator: the campaign's first node builds one from the brief's protocol and every candidate is measured against it | `kapso build --brief kapso-brief.md --data data --repo repo --output campaign --detach` | not recorded yet |
| [pipeline-dbt](pipeline-dbt/README.md) | 30 | a dbt pipeline on DuckDB whose exports carry defects: the data contract made to hold at the staging layer; the third domain a harness author writes for | `kapso build --brief kapso-brief.md --evaluation evaluation --data data --repo repo --output campaign --detach` | not yet run as a build |
| [trading](trading/README.md) | 75 | a harness on the build: a setup session, pages and hooks in every session, verification per attempt | `kapso build --brief kapso-brief.md --evaluation evaluation --data data --repo repo --output campaign --detach` | recorded in its README |
| [full-loop](full-loop/README.md) | 45, then the loop | build, the review, learn, build again served the lesson, deploy: a Kaggle tabular model end to end | `python run.py` | not recorded yet |
| [prompt-harness](prompt-harness/README.md) | 60 | the prompt as the thing under test, scored on AIME problems | `kapso build --brief kapso-brief.md --repo repo --output campaign --detach` | not recorded yet; runs today, its evaluator inside `repo/`, its harness a follow-up |
| [agent-scaffold](agent-scaffold/README.md) | 60 | a vision-language extractor, scored against ground-truth tables | `kapso build --brief kapso-brief.md --repo repo --output campaign --detach` | not recorded yet; runs today, its evaluator inside `repo/`, its harness a follow-up |
| [torch-performance](torch-performance/README.md) | 60, CUDA | a faster module under a numerical tolerance | `kapso build --brief kapso-brief.md --repo repo --output campaign --detach` | not recorded yet; runs today, its evaluator inside `repo/`, its harness a follow-up |
| [gpu-kernels](gpu-kernels/README.md) | 90, CUDA | multi-head attention, flash attention or Triton | `kapso build --brief kapso-brief.md --repo repo --output campaign --detach` | not recorded yet; runs today, its evaluator inside `repo/`, its harness a follow-up |

Each `run.py` is at most twelve lines of code: read the brief, build, print
the result; `full-loop` spends its twelve on the whole loop.
`tests/test_examples_drivers.py` holds every driver to that and to the
keywords the API exposes, so a driver cannot drift from the interface it
shows.

The examples `tests/semantic/suite.json` names (churn, fraud, the empty
wish, trading, pipeline-dbt) are also the fixtures of the semantic
production tests (`tests/semantic/README.md`): the same directories run on
a fresh machine from the released package, so a page that quotes one
quotes something that ran.
