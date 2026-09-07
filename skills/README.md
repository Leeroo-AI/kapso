# Skills for coding agents

One folder per coding agent, holding everything that agent needs. The
`kapso` skill teaches an agent to operate Kapso from a session: verify the
install, launch and follow a campaign, answer the inbox, resume, learn,
ingest knowledge, deploy.

| Folder | Agent | Discovered from |
|---|---|---|
| `claude-code/kapso/` | Claude Code | `.claude/skills/kapso` (project) or `~/.claude/skills/kapso` (personal) |
| `codex/kapso/` | Codex CLI | `.agents/skills/kapso` (repo, walking up to the git root) or `~/.agents/skills/kapso` (user); `agents/openai.yaml` decorates the `/skills` list |
| `opencode/kapso/` | OpenCode | `.opencode/skills/kapso` (project) or `~/.config/opencode/skills/kapso` (global); OpenCode also reads the Claude and Codex locations |

The copies are deliberately separate files, not symlinks, so each can diverge
as its agent's conventions do. Today they differ in one place: Codex's shell
ends a plain `nohup … &` child when the call returns, so the Codex copy
launches and resumes campaigns with `setsid -f`; the other two use `nohup`.
Their test record is `docs/plans/coding-agent-skill-findings.md`.

## Install

From a checkout, into the project you want the agent to use it in:

```bash
# Claude Code
mkdir -p .claude/skills && ln -s /path/to/kapso/skills/claude-code/kapso .claude/skills/kapso

# Codex
mkdir -p .agents/skills && ln -s /path/to/kapso/skills/codex/kapso .agents/skills/kapso

# OpenCode
mkdir -p .opencode/skills && ln -s /path/to/kapso/skills/opencode/kapso .opencode/skills/kapso
```

OpenCode also reads `.claude/skills` and `.agents/skills`, so in a project
that already has one of the other two symlinks it needs none of its own.
