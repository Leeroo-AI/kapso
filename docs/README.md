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

**Getting started**

- [What is Kapso](https://docs.leeroo.com/docs) — `docs/index.mdx`
- [Install Kapso and verify your setup with kapso doctor](https://docs.leeroo.com/docs/installation) — `docs/installation.mdx`
- [Run your first Kapso campaign in under ten minutes](https://docs.leeroo.com/docs/quickstart) — `docs/quickstart.mdx`
- [Docs in your coding agent](https://docs.leeroo.com/docs/agent-access) — `docs/agent-access.mdx`
- [Kapso skill for coding agents](https://docs.leeroo.com/docs/coding-agent-skills) — `docs/coding-agent-skills.mdx`

**Evolve system**

- [Overview](https://docs.leeroo.com/docs/evolve/overview) — `docs/evolve/overview.mdx`
- [Architecture](https://docs.leeroo.com/docs/evolve/architecture) — `docs/evolve/architecture.mdx`
- [Execution flow](https://docs.leeroo.com/docs/evolve/execution-flow) — `docs/evolve/execution-flow.mdx`
- [Resuming runs](https://docs.leeroo.com/docs/evolve/resuming-runs) — `docs/evolve/resuming-runs.mdx`
- [Inbox](https://docs.leeroo.com/docs/evolve/inbox) — `docs/evolve/inbox.mdx`
- [External evaluation](https://docs.leeroo.com/docs/evolve/external-evaluation) — `docs/evolve/external-evaluation.mdx`
- [Evaluation integrity](https://docs.leeroo.com/docs/evolve/evaluation-integrity) — `docs/evolve/evaluation-integrity.mdx`

*Components*

- [Orchestrator](https://docs.leeroo.com/docs/evolve/orchestrator) — `docs/evolve/orchestrator.mdx`
- [Search strategies](https://docs.leeroo.com/docs/evolve/search-strategies) — `docs/evolve/search-strategies.mdx`
- [Parent selection](https://docs.leeroo.com/docs/evolve/parent-selection) — `docs/evolve/parent-selection.mdx`
- [MCP gates](https://docs.leeroo.com/docs/evolve/mcp-gates) — `docs/evolve/mcp-gates.mdx`
- [Model routing](https://docs.leeroo.com/docs/evolve/model-routing-retries) — `docs/evolve/model-routing-retries.mdx`
- [Coding agents](https://docs.leeroo.com/docs/evolve/coding-agents) — `docs/evolve/coding-agents.mdx`
- [Feedback generator](https://docs.leeroo.com/docs/evolve/feedback-generator) — `docs/evolve/feedback-generator.mdx`
- [Experiment lifecycle](https://docs.leeroo.com/docs/evolve/experiment-lifecycle) — `docs/evolve/experiment-lifecycle.mdx`
- [Repo memory](https://docs.leeroo.com/docs/evolve/repo-memory) — `docs/evolve/repo-memory.mdx`

**Knowledge graph**

- [Knowledge graph](https://docs.leeroo.com/docs/knowledge/overview) — `docs/knowledge/overview.mdx`
- [Learning pipeline](https://docs.leeroo.com/docs/knowledge/learning-pipeline) — `docs/knowledge/learning-pipeline.mdx`
- [Search backends](https://docs.leeroo.com/docs/knowledge/search-backends) — `docs/knowledge/search-backends.mdx`

**Trajectory learning**

- [Overview](https://docs.leeroo.com/docs/learning/overview) — `docs/learning/overview.mdx`
- [The lesson bank](https://docs.leeroo.com/docs/learning/bank) — `docs/learning/bank.mdx`
- [The pipeline](https://docs.leeroo.com/docs/learning/pipeline) — `docs/learning/pipeline.mdx`
- [Grading](https://docs.leeroo.com/docs/learning/graders) — `docs/learning/graders.mdx`
- [Serving](https://docs.leeroo.com/docs/learning/serving) — `docs/learning/serving.mdx`
- [Development regime](https://docs.leeroo.com/docs/learning/development) — `docs/learning/development.mdx`

**Research**

- [Deep web research](https://docs.leeroo.com/docs/research/overview) — `docs/research/overview.mdx`

**Deployment**

- [Deployment](https://docs.leeroo.com/docs/deployment/overview) — `docs/deployment/overview.mdx`
- [Deployment architecture](https://docs.leeroo.com/docs/deployment/architecture) — `docs/deployment/architecture.mdx`
- [Deployment strategies](https://docs.leeroo.com/docs/deployment/strategies) — `docs/deployment/strategies.mdx`
- [Adding strategies](https://docs.leeroo.com/docs/deployment/adding-strategies) — `docs/deployment/adding-strategies.mdx`

**Benchmarks**

- [IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) — `docs/benchmarks/ioai-2026.mdx`
- [MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) — `docs/benchmarks/mle-bench.mdx`
- [ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) — `docs/benchmarks/ale-bench.mdx`
- [RelBench](https://docs.leeroo.com/docs/benchmarks/relbench) — `docs/benchmarks/relbench.mdx`

**Reference**

- [CLI](https://docs.leeroo.com/docs/reference/cli) — `docs/reference/cli.mdx`
- [Python API](https://docs.leeroo.com/docs/reference/kapso-api) — `docs/reference/kapso-api.mdx`
- [Deployment API](https://docs.leeroo.com/docs/deployment/api-reference) — `docs/deployment/api-reference.mdx`
- [Configuration](https://docs.leeroo.com/docs/reference/configuration) — `docs/reference/configuration.mdx`
