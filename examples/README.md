# Examples

Every example is a task: a brief (`kapso-brief.md`), a starting repository
(`repo/`, or the files that assemble it), the data or how to get it, and
`run.py`, the same build the one command runs. `churn-first-build` and
`trading` carry a seeded data generator and a protected evaluator
(`evaluation/`); `fraud-no-evaluator` carries the generator and lets the
campaign build the evaluator from the brief's protocol. `dbt-snowflake` and
`support-agent-harness` are full-size: a warehouse on Snowflake and a model
on a GPU, each with its brief, its evaluation and a README that says what to
bring and how to set it up. `torch-performance` and `gpu-kernels` run today
with their evaluator inside the repository, where the judge protects it by
rule; a harness for each is a follow-up. `harnesses/` holds the harnesses
the trading, dbt and support-agent examples build on. Install Kapso and log
the coding-agent CLIs in first (`kapso doctor build` must pass), then run
from the example's directory.

Every example runs the experimentation; `end-to-end`, `dbt-snowflake` and
`support-agent-harness` add the deployment and the learning, and `trading`,
`dbt-snowflake` and `support-agent-harness` run on a harness, the domain
adaptation.

| Example | Minutes | What it shows | The one command | Result |
|---|---|---|---|---|
| [dbt-snowflake](dbt-snowflake/README.md) | up to 720, Snowflake | a backlog of seventeen tickets on a dbt warehouse, worked on the data-engineering harness until every ticket's check passes; then one pull request per ticket and a release to production, through the harness's own delivery targets | `kapso build --brief kapso-brief.md --repo repo --evaluation evaluation --output campaign --config repo/kapso-config.yaml --detach` | none quoted: its README is the setup |
| [support-agent-harness](support-agent-harness/README.md) | up to 720, an 80 GB GPU | the harness of a customer support agent optimized on the agent-engineering harness, so that a locally served model resolves the conversations as well as the frontier model it replaces; then a Docker deployment and a learn run | `kapso build --brief kapso-brief.md --repo repo --evaluation evaluation --data data --output campaign --harness ../harnesses/agent-engineering --config repo/kapso-config.yaml --detach` | none quoted: its README is the setup |
| [churn-first-build](churn-first-build/README.md) | 45 | your first build: an evaluator present, a baseline, a target, the record | `kapso build --brief kapso-brief.md --evaluation evaluation --data data --repo repo --output campaign --detach` | recorded in its README |
| [fraud-no-evaluator](fraud-no-evaluator/README.md) | 45 | no evaluator: the campaign's first node builds one from the brief's protocol and every candidate is measured against it | `kapso build --brief kapso-brief.md --data data --repo repo --output campaign --detach` | not recorded yet |
| [trading](trading/README.md) | 75 | a harness on the build: a setup session, pages and hooks in every session, verification per attempt | `kapso build --brief kapso-brief.md --evaluation evaluation --data data --repo repo --output campaign --detach` | recorded in its README |
| [end-to-end](end-to-end/README.md) | 45, then the rest | build, the review, learn, build again served the lesson, deploy: a Kaggle tabular model end to end | `python run.py` | not recorded yet |
| [torch-performance](torch-performance/README.md) | 60, CUDA | a faster module under a numerical tolerance | `kapso build --brief kapso-brief.md --repo repo --output campaign --detach` | not recorded yet; runs today, its evaluator inside `repo/`, its harness a follow-up |
| [gpu-kernels](gpu-kernels/README.md) | 90, CUDA | multi-head attention, flash attention or Triton | `kapso build --brief kapso-brief.md --repo repo --output campaign --detach` | not recorded yet; runs today, its evaluator inside `repo/`, its harness a follow-up |

Each `run.py` is at most twelve lines of code: read the brief, build, print
the result; `end-to-end` spends its twelve on the whole sequence.
`tests/test_examples_drivers.py` holds every driver to that and to the
keywords the API exposes, so a driver cannot drift from the interface it
shows.

The examples `tests/semantic/suite.json` names (churn, fraud, the empty
wish, trading) are also the fixtures of the semantic production tests
(`tests/semantic/README.md`): the same directories run on a fresh machine
from the released package, so a page that quotes one quotes something that
ran.
