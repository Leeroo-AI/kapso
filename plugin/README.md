# The Kapso plugin

Kapso is an engine that runs experiments, deploys the best solution and
learns from every run: it proposes candidate solutions from previous
experiments and its bank of lessons, has coding agents implement them on
branches of their own, measures each against your evaluation, keeps the
best, deploys it on your stack and learns from the run. This plugin is
how you meet it from the coding agent you already work in.
It carries the skills (the procedures for setting up, stating an
objective with its evaluation, operating a campaign, reviewing a result,
authoring a harness), the commands that start each one, and the MCP
server that gives your session Kapso's own tools (web research, the
experience you have banked, a campaign's history).

## Install

```text
claude plugin marketplace add Leeroo-AI/kapso   &&  claude plugin install kapso@kapso
codex plugin marketplace add Leeroo-AI/kapso    &&  codex plugin add kapso@kapso
```

Then, in a session, `/kapso:setup` (Claude Code) or `$kapso-setup`
(Codex) installs the package and the coding-agent CLIs it drives, runs
the logins and checks the machine with `kapso doctor`. The first
campaign is `/kapso:build <what you want>`.

## Layout

```text
.claude-plugin/plugin.json   Claude Code's manifest
.codex-plugin/plugin.json    Codex's manifest
.cursor-plugin/plugin.json   Cursor's manifest
hosts/<host>/mcp.json        the MCP entry each host runs (Claude Code: the launcher below; Codex, Cursor: kapso-mcp on PATH)
scripts/kapso-mcp            the launcher: the installed kapso-mcp, else the pinned release through uvx or pipx
commands/                    /kapso:setup build watch inbox resume review learn deploy harness experience
skills/                      kapso, kapso-setup, kapso-build, kapso-evaluation, kapso-harness, kapso-review
evals/                       the trigger suite
```

The plugin's version is the package's; the launcher pins it. The page:
https://docs.leeroo.com/docs/coding-agent-skills.
