# Editing these docs

[docs.leeroo.com](https://docs.leeroo.com/docs) is built and served by Mintlify.
Push to `main` and the site rebuilds itself — there is no deploy step to run and
no machine of ours in the path. The only infrastructure we own is the DNS record
pointing `docs.leeroo.com` at Mintlify.

## Change a page

1. Edit the `.mdx` file.
2. Open a PR. CI runs `mint validate` and `mint broken-links` against it.
3. Merge. Mintlify rebuilds `main`; the change is live in a minute or two.

## Add a page

Create the `.mdx` file **and** add its path to `navigation` in `/docs.json`.

Both halves matter. A file that is not in the navigation is not published —
Mintlify serves 404 for it. A navigation entry with no file behind it puts a
dead link in the sidebar, which is exactly what CI is there to catch.

## Preview locally

```bash
npx mint@4 dev            # http://localhost:3000
npx mint@4 validate       # what CI runs
npx mint@4 broken-links   # what CI runs
```

## Worth knowing

- **CI fails on warnings, not just errors.** A page referenced in `docs.json`
  that does not exist is only a warning, and it is the bug this gate exists for.
- **Drafts do not belong in `docs/`.** Everything here is parsed by the build.
  Internal notes live under `docs/plans/` and `docs/research/*.md`, which
  `/.mintignore` excludes; anything else new is fair game for the build.
- **Patterns in `/.mintignore` must be anchored.** An unanchored `benchmarks/`
  also matches `docs/benchmarks/` and silently unpublishes those pages.
- **There are no PR preview deployments** on the current plan. CI passing is the
  only signal you get before a change is live, so read it.

## Pages

Every published page with its rendered address, grouped as in the sidebar. Kept here so the folder listing on GitHub points at the site.

**Start**

- [What is Kapso](https://docs.leeroo.com/docs) — `docs/index.mdx`
- [Install Kapso and verify your setup with kapso doctor](https://docs.leeroo.com/docs/installation) — `docs/installation.mdx`
- [Your first build](https://docs.leeroo.com/docs/first-build) — `docs/first-build.mdx`
- [Kapso in your coding agent](https://docs.leeroo.com/docs/coding-agent-skills) — `docs/coding-agent-skills.mdx`

**Build**

- [The brief](https://docs.leeroo.com/docs/build/brief) — `docs/build/brief.mdx`
- [Evaluation](https://docs.leeroo.com/docs/build/evaluation) — `docs/build/evaluation.mdx`
- [Run and watch](https://docs.leeroo.com/docs/build/run-and-watch) — `docs/build/run-and-watch.mdx`
- [Inbox](https://docs.leeroo.com/docs/build/inbox) — `docs/build/inbox.mdx`
- [Review the result](https://docs.leeroo.com/docs/build/review) — `docs/build/review.mdx`
- [Read the record](https://docs.leeroo.com/docs/build/trace) — `docs/build/trace.mdx`
- [Resume and budgets](https://docs.leeroo.com/docs/build/resume-and-budgets) — `docs/build/resume-and-budgets.mdx`

**The experimentation loop**

- [The experimentation loop](https://docs.leeroo.com/docs/evolve/overview) — `docs/evolve/overview.mdx`
- [Architecture](https://docs.leeroo.com/docs/evolve/architecture) — `docs/evolve/architecture.mdx`
- [Execution flow](https://docs.leeroo.com/docs/evolve/execution-flow) — `docs/evolve/execution-flow.mdx`
- [Evaluation integrity](https://docs.leeroo.com/docs/evolve/evaluation-integrity) — `docs/evolve/evaluation-integrity.mdx`
- [External evaluation](https://docs.leeroo.com/docs/evolve/external-evaluation) — `docs/evolve/external-evaluation.mdx`
- [Orchestrator](https://docs.leeroo.com/docs/evolve/orchestrator) — `docs/evolve/orchestrator.mdx`
- [Search strategies](https://docs.leeroo.com/docs/evolve/search-strategies) — `docs/evolve/search-strategies.mdx`
- [Parent selection](https://docs.leeroo.com/docs/evolve/parent-selection) — `docs/evolve/parent-selection.mdx`
- [Cost-aware selection](https://docs.leeroo.com/docs/evolve/cost-aware-selection) — `docs/evolve/cost-aware-selection.mdx`
- [MCP gates](https://docs.leeroo.com/docs/evolve/mcp-gates) — `docs/evolve/mcp-gates.mdx`
- [Model routing](https://docs.leeroo.com/docs/evolve/model-routing-retries) — `docs/evolve/model-routing-retries.mdx`
- [Coding agents](https://docs.leeroo.com/docs/evolve/coding-agents) — `docs/evolve/coding-agents.mdx`
- [Feedback generator](https://docs.leeroo.com/docs/evolve/feedback-generator) — `docs/evolve/feedback-generator.mdx`
- [Experiment lifecycle](https://docs.leeroo.com/docs/evolve/experiment-lifecycle) — `docs/evolve/experiment-lifecycle.mdx`
- [Repo memory](https://docs.leeroo.com/docs/evolve/repo-memory) — `docs/evolve/repo-memory.mdx`

**Learn**

- [What learn produces](https://docs.leeroo.com/docs/learn/what-learn-produces) — `docs/learn/what-learn-produces.mdx`
- [Development mode](https://docs.leeroo.com/docs/learn/development-mode) — `docs/learn/development-mode.mdx`

**Experience**

- [What experience is](https://docs.leeroo.com/docs/experience) — `docs/experience/index.mdx`
- [Your bank on a build](https://docs.leeroo.com/docs/experience/use) — `docs/experience/use.mdx`
- [Add a bank, Leeroopedia, an index](https://docs.leeroo.com/docs/experience/add) — `docs/experience/add.mdx`
- [Share a bank](https://docs.leeroo.com/docs/experience/share) — `docs/experience/share.mdx`

**Deploy**

- [Targets](https://docs.leeroo.com/docs/deploy/targets) — `docs/deploy/targets.mdx`
- [Write a target skill](https://docs.leeroo.com/docs/deploy/write-a-target-skill) — `docs/deploy/write-a-target-skill.mdx`

**Harnesses**

- [What a harness is](https://docs.leeroo.com/docs/harnesses) — `docs/harnesses/index.mdx`
- [Use a harness](https://docs.leeroo.com/docs/harnesses/use) — `docs/harnesses/use.mdx`
- [Predictive modelling](https://docs.leeroo.com/docs/harnesses/bundled/predictive-modelling) — `docs/harnesses/bundled/predictive-modelling.mdx`
- [Competitive programming](https://docs.leeroo.com/docs/harnesses/bundled/competitive-programming) — `docs/harnesses/bundled/competitive-programming.mdx`
- [Write a harness](https://docs.leeroo.com/docs/harnesses/write) — `docs/harnesses/write.mdx`
- [The trading example](https://docs.leeroo.com/docs/harnesses/example-trading) — `docs/harnesses/example-trading.mdx`

**Benchmarks**

- [IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) — `docs/benchmarks/ioai-2026.mdx`
- [MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) — `docs/benchmarks/mle-bench.mdx`
- [ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) — `docs/benchmarks/ale-bench.mdx`
- [RelBench](https://docs.leeroo.com/docs/benchmarks/relbench) — `docs/benchmarks/relbench.mdx`

**Reference**

- [CLI](https://docs.leeroo.com/docs/reference/cli) — `docs/reference/cli.mdx`
- [Python API](https://docs.leeroo.com/docs/reference/kapso-api) — `docs/reference/kapso-api.mdx`
- [Brief format](https://docs.leeroo.com/docs/reference/brief-format) — `docs/reference/brief-format.mdx`
- [Harness](https://docs.leeroo.com/docs/reference/harness) — `docs/reference/harness.mdx`
- [Experience](https://docs.leeroo.com/docs/reference/experience) — `docs/reference/experience.mdx`
- [Configuration](https://docs.leeroo.com/docs/reference/configuration) — `docs/reference/configuration.mdx`
